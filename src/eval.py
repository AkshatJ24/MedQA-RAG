"""
eval.py — RAGAS Evaluation & A/B Testing
Compares: RAG System (chain.py) vs Baseline LLM (no retrieval)
Metric:   Faithfulness only (LLM-as-judge, no embeddings needed)
Dataset:  Held-out sample from keivalya/MedQuad-MedicalQnADataset
"""

import os
import json
import math
import time
import logging
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from datasets import load_dataset, Dataset

# RAGAS
from ragas import evaluate
from ragas.metrics import faithfulness
from ragas.llms import llm_factory
from ragas.run_config import RunConfig

# Groq via OpenAI-compatible client
from openai import OpenAI as GroqClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Our RAG chain
from chain import HealthcareQAChain

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY")
EVAL_SAMPLE_SIZE  = 10
RANDOM_SEED       = 42
RESULTS_DIR       = "data/eval_results"
SLEEP_BETWEEN     = 5   # seconds between LLM calls to avoid rate limits

EVAL_QTYPES = [
    "symptoms",
    "treatment",
    "prevention",
    "causes",
    "exams and tests",
]
SAMPLES_PER_QTYPE = EVAL_SAMPLE_SIZE // len(EVAL_QTYPES)


# ─────────────────────────────────────────────
# STEP 1: Build Evaluation Dataset
# ─────────────────────────────────────────────
def build_eval_dataset() -> pd.DataFrame:
    """
    Samples questions evenly across 5 qtypes from MedQuAD.
    Ground truth = NIH verified answer from the dataset.
    """
    log.info("Loading MedQuAD dataset for evaluation sampling...")
    dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
    df = dataset.to_pandas()

    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Question": "question", "Answer": "ground_truth"})
    df["qtype"] = df["qtype"].str.strip().str.lower()
    df = df.dropna(subset=["question", "ground_truth"])
    df = df[df["ground_truth"].str.len() > 50]

    sampled_frames = []
    for qtype in EVAL_QTYPES:
        subset = df[df["qtype"] == qtype]
        n = min(SAMPLES_PER_QTYPE, len(subset))
        sampled_frames.append(subset.sample(n=n, random_state=RANDOM_SEED))

    eval_df = pd.concat(sampled_frames).reset_index(drop=True)
    log.info(f"Eval dataset ready: {len(eval_df)} questions across {len(EVAL_QTYPES)} qtypes.")
    return eval_df


# ─────────────────────────────────────────────
# STEP 2A: Run RAG System
# ─────────────────────────────────────────────
def run_rag_system(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs each question through the full RAG pipeline.
    Uses llama-3.1-8b-instant to save daily token quota.
    """
    log.info("Running RAG system on eval dataset...")
    chain = HealthcareQAChain()

    # Use lighter model for eval to preserve daily token quota.
    # The Streamlit app (chain.py) still uses llama-3.3-70b-versatile.
    chain.llm = ChatOpenAI(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.2,
        max_tokens=512,
    )

    rag_answers  = []
    rag_contexts = []
    rag_fallback = []

    for i, row in eval_df.iterrows():
        question = row["question"]
        log.info(f"  [{i+1}/{len(eval_df)}] {question[:70]}...")

        chain.reset_memory()
        result = chain.ask(question)
        time.sleep(SLEEP_BETWEEN)

        rag_answers.append(result["answer"])
        rag_fallback.append(result["fallback"])

        if not result["fallback"]:
            docs     = chain.vectorstore.similarity_search(question, k=3)
            contexts = [doc.page_content for doc in docs]
        else:
            contexts = ["No context retrieved — fallback triggered."]

        rag_contexts.append(contexts)

    eval_df["rag_answer"]   = rag_answers
    eval_df["rag_contexts"] = rag_contexts
    eval_df["rag_fallback"] = rag_fallback

    fallback_count = sum(rag_fallback)
    log.info(f"RAG done. Fallbacks triggered: {fallback_count}/{len(eval_df)}")
    return eval_df


# ─────────────────────────────────────────────
# STEP 2B: Run Baseline LLM (no retrieval)
# ─────────────────────────────────────────────
BASELINE_PROMPT = """You are a medical assistant. Answer the following medical \
question based on your general knowledge. Be concise and factual.

Question: {question}

Answer:"""

def run_baseline_llm(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs each question through Groq with NO retrieval context.
    This is the control group — shows raw LLM performance without RAG.
    """
    log.info("Running baseline LLM (no retrieval) on eval dataset...")
    llm = ChatOpenAI(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.2,
        max_tokens=512,
    )

    baseline_answers = []
    for i, row in eval_df.iterrows():
        question = row["question"]
        log.info(f"  [{i+1}/{len(eval_df)}] Baseline: {question[:70]}...")

        prompt   = BASELINE_PROMPT.format(question=question)
        response = llm.invoke([HumanMessage(content=prompt)])
        baseline_answers.append(response.content.strip())
        time.sleep(SLEEP_BETWEEN)

    eval_df["baseline_answer"] = baseline_answers
    log.info("Baseline LLM done.")
    return eval_df


# ─────────────────────────────────────────────
# RAGAS JUDGE SETUP
# Uses llm_factory (RAGAS native) with Groq.
# Only faithfulness is evaluated — it needs only
# the LLM, no embeddings, no async issues.
# ─────────────────────────────────────────────
def get_ragas_judge():
    groq_client = GroqClient(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    judge_llm = llm_factory(
        model="llama-3.1-8b-instant",
        client=groq_client,
    )
    run_cfg = RunConfig(
        max_retries=3,
        max_wait=30,
        timeout=90,
    )
    return judge_llm, run_cfg


# ─────────────────────────────────────────────
# STEP 3: Score with RAGAS (faithfulness only)
# ─────────────────────────────────────────────
def score_with_ragas(eval_df: pd.DataFrame, answer_col: str) -> dict:
    """
    Scores answers using RAGAS faithfulness metric.

    Faithfulness: measures whether every statement in the generated
    answer can be inferred from the retrieved context chunks.
    Score of 1.0 = fully grounded, 0.0 = completely hallucinated.

    Requires only:
      - question     : user query
      - answer       : generated answer to score
      - contexts     : list of retrieved chunks (truncated for token efficiency)
    """
    log.info(f"Scoring with RAGAS faithfulness — column: '{answer_col}'")

    # Truncate inputs to keep scoring prompts short enough for 8b model
    def truncate(text: str, max_chars: int = 500) -> str:
        text = str(text)
        return text[:max_chars] + "..." if len(text) > max_chars else text

    def truncate_contexts(ctx_list: list, max_chars: int = 350) -> list:
        # Max 3 chunks, each truncated — prevents prompt overflow
        return [c[:max_chars] for c in ctx_list[:3]]

    ragas_data = {
        "question": eval_df["question"].tolist(),
        "answer":   [truncate(a) for a in eval_df[answer_col].tolist()],
        "contexts": [truncate_contexts(c) for c in eval_df["rag_contexts"].tolist()],
    }

    ragas_dataset = Dataset.from_dict(ragas_data)
    judge_llm, run_cfg = get_ragas_judge()

    result = evaluate(
        dataset=ragas_dataset,
        metrics=[faithfulness],
        llm=judge_llm,
        run_config=run_cfg,
    )

    def safe_score(val) -> float:
        """Handles float, list, or NaN returned by RAGAS safely."""
        if isinstance(val, list):
            valid = [v for v in val if v is not None and not (isinstance(v, float) and math.isnan(v))]
            return round(sum(valid) / len(valid), 4) if valid else 0.0
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return 0.0
        return round(float(val), 4)

    scores = {
        "faithfulness":       safe_score(result["faithfulness"]),
        "answer_correctness": "N/A",  # skipped — requires async embeddings
        "context_precision":  "N/A",  # skipped — requires async embeddings
    }

    log.info(f"RAGAS scores ({answer_col}): faithfulness = {scores['faithfulness']}")
    return scores


# ─────────────────────────────────────────────
# STEP 4: A/B Report
# ─────────────────────────────────────────────
def print_ab_report(rag_scores: dict, baseline_scores: dict, eval_df: pd.DataFrame):
    sep = "=" * 62

    print(f"\n{sep}")
    print("  A/B EVALUATION REPORT — Healthcare FAQ Assistant")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Metric: Faithfulness (LLM-as-judge via Groq)")
    print(sep)

    rag_f  = rag_scores["faithfulness"]
    base_f = baseline_scores["faithfulness"]

    # Handle N/A safely
    if isinstance(rag_f, float) and isinstance(base_f, float):
        delta = rag_f - base_f
        arrow = "↑" if delta >= 0 else "↓"
        delta_str = f"{arrow} {abs(delta):.4f}"
    else:
        delta_str = "N/A"

    print(f"\n{'Metric':<25} {'RAG System':>12} {'Baseline LLM':>14} {'Delta':>12}")
    print("-" * 65)
    print(f"{'faithfulness':<25} {str(rag_f):>12} {str(base_f):>14} {delta_str:>12}")
    print(f"{'answer_correctness':<25} {'N/A':>12} {'N/A':>14} {'—':>12}")
    print(f"{'context_precision':<25} {'N/A':>12} {'N/A':>14} {'—':>12}")

    print(f"\n{sep}")
    print("  PER-QTYPE BREAKDOWN (RAG System)")
    print(sep)

    qtype_stats = eval_df.groupby("qtype").agg(
        count     = ("question", "count"),
        fallbacks = ("rag_fallback", "sum"),
        avg_len   = ("rag_answer", lambda x: round(x.str.len().mean(), 0)),
    ).reset_index()

    print(f"\n{'QType':<22} {'Count':>7} {'Fallbacks':>10} {'Avg Answer Len':>16}")
    print("-" * 58)
    for _, row in qtype_stats.iterrows():
        print(f"{row['qtype']:<22} {int(row['count']):>7} {int(row['fallbacks']):>10} {int(row['avg_len']):>16}")

    print(f"\n{sep}")
    print(f"  Total evaluated : {len(eval_df)} questions")
    print(f"  Total fallbacks : {int(eval_df['rag_fallback'].sum())}")
    print(f"\n  Faithfulness measures whether RAG answers are grounded")
    print(f"  in retrieved NIH context — higher is better.")
    print(f"  RAG ({rag_f}) vs Baseline ({base_f}) shows the impact")
    print(f"  of retrieval augmentation on answer reliability.")
    print(sep)


# ─────────────────────────────────────────────
# STEP 5: Save Results
# ─────────────────────────────────────────────
def save_results(eval_df: pd.DataFrame, rag_scores: dict, baseline_scores: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(RESULTS_DIR, f"eval_results_{timestamp}.csv")
    eval_df[[
        "question", "qtype", "ground_truth",
        "rag_answer", "baseline_answer", "rag_fallback"
    ]].to_csv(csv_path, index=False)
    log.info(f"Detailed results saved → {csv_path}")

    summary = {
        "timestamp":       timestamp,
        "eval_size":       len(eval_df),
        "metric":          "faithfulness only",
        "rag_scores":      rag_scores,
        "baseline_scores": baseline_scores,
        "faithfulness_delta": (
            round(rag_scores["faithfulness"] - baseline_scores["faithfulness"], 4)
            if isinstance(rag_scores["faithfulness"], float)
            and isinstance(baseline_scores["faithfulness"], float)
            else "N/A"
        )
    }
    json_path = os.path.join(RESULTS_DIR, f"scores_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Score summary saved   → {json_path}")

    return csv_path, json_path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_evaluation():
    log.info("Starting evaluation pipeline...")

    eval_df         = build_eval_dataset()
    eval_df         = run_rag_system(eval_df)
    eval_df         = run_baseline_llm(eval_df)
    rag_scores      = score_with_ragas(eval_df, answer_col="rag_answer")
    baseline_scores = score_with_ragas(eval_df, answer_col="baseline_answer")

    print_ab_report(rag_scores, baseline_scores, eval_df)

    csv_path, json_path = save_results(eval_df, rag_scores, baseline_scores)

    print(f"\n✅ Evaluation complete.")
    print(f"   Detailed CSV → {csv_path}")
    print(f"   Score JSON   → {json_path}")

    return rag_scores, baseline_scores, eval_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAGAS Evaluation — Healthcare FAQ Assistant")
    parser.add_argument(
        "--sample-size", type=int, default=EVAL_SAMPLE_SIZE,
        help=f"Number of questions to evaluate (default: {EVAL_SAMPLE_SIZE})"
    )
    args = parser.parse_args()

    EVAL_SAMPLE_SIZE  = args.sample_size
    SAMPLES_PER_QTYPE = max(1, EVAL_SAMPLE_SIZE // len(EVAL_QTYPES))

    run_evaluation()