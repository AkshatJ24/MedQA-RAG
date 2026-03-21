"""
app.py — Streamlit UI for Healthcare FAQ Assistant
Dark mode, fixed layout, Groq + local embeddings
"""

import streamlit as st
import json
import os
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare FAQ Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# FORCE DARK MODE VIA CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [data-testid="stApp"], .stApp,
    [data-testid="stAppViewContainer"], .main,
    [data-testid="stMainBlockContainer"],
    section.main > div {
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background-color: #0a1628 !important;
        border-right: 1px solid #21262d !important;
    }
    [data-testid="stSidebar"] * { color: #8b949e !important; }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] strong { color: #e6edf3 !important; }
    [data-testid="stSidebar"] hr { border-color: #21262d !important; }

    p, span, div, label, li { color: #e6edf3; }
    h1, h2, h3, h4 { color: #e6edf3 !important; }

    .stButton > button {
        background: #1f6feb !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        transition: background 0.2s !important;
    }
    .stButton > button:hover { background: #388bfd !important; }

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: #161b22 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #1f6feb !important;
        box-shadow: 0 0 0 3px rgba(31,111,235,0.15) !important;
    }

    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        border-radius: 8px !important;
    }

    [data-testid="stDataFrame"], .stDataFrame {
        background: #161b22 !important;
        border: 1px solid #21262d !important;
        border-radius: 10px !important;
    }

    [data-testid="stForm"] { background: transparent !important; border: none !important; }

    .stProgress > div > div > div { background: #1f6feb !important; }

    .app-header {
        background: linear-gradient(135deg, #0d1f3c 0%, #1f3a6e 100%);
        border: 1px solid rgba(31,111,235,0.25);
        border-radius: 14px;
        padding: 24px 32px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .app-header h1 {
        font-family: 'DM Serif Display', serif;
        color: #e6edf3 !important;
        font-size: 1.9rem;
        margin: 0;
    }
    .app-header p { color: #8b949e !important; margin: 4px 0 0 0; font-size: 0.88rem; }

    .chat-user {
        background: #1f3a6e;
        color: #e6edf3;
        border-radius: 16px 16px 4px 16px;
        padding: 12px 16px;
        margin: 6px 0 6px 60px;
        font-size: 0.93rem;
        line-height: 1.6;
        border: 1px solid rgba(31,111,235,0.25);
    }
    .chat-assistant {
        background: #161b22;
        color: #e6edf3;
        border-radius: 16px 16px 16px 4px;
        padding: 14px 18px;
        margin: 6px 60px 6px 0;
        font-size: 0.93rem;
        line-height: 1.7;
        border: 1px solid #21262d;
        border-left: 3px solid #1f6feb;
    }
    .chat-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin-bottom: 4px;
        color: #484f58;
    }

    .fallback-box {
        background: #2a1f0f;
        border: 1px solid #d29922;
        border-left: 4px solid #d29922;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 60px 6px 0;
        color: #e3b341;
        font-size: 0.9rem;
    }

    .source-card {
        background: #161b22;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        border: 1px solid #21262d;
        border-left: 3px solid #1f6feb;
        font-size: 0.83rem;
        color: #8b949e;
    }
    .source-card b { color: #e6edf3; }
    .source-qtype {
        display: inline-block;
        background: #1f3a6e;
        color: #79c0ff;
        border: 1px solid rgba(31,111,235,0.3);
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .metric-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-value {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        color: #79c0ff;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #484f58;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-top: 6px;
    }
    .metric-delta-pos { color: #3fb950; font-size: 0.82rem; font-weight: 600; margin-top: 4px; }
    .metric-delta-neg { color: #f85149; font-size: 0.82rem; font-weight: 600; margin-top: 4px; }
    .metric-baseline  { color: #6e7681; font-size: 2rem; font-family: 'DM Serif Display', serif; }

    .disclaimer {
        background: #1c2a1c;
        border: 1px solid rgba(46,160,67,0.2);
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.8rem;
        color: #7ee787;
        margin-top: 12px;
    }

    .empty-state { text-align: center; padding: 40px 0; }
    .empty-state .icon { font-size: 2.5rem; }
    .empty-state p { font-size: 0.88rem; margin-top: 10px; color: #484f58; }

    hr { border-color: #21262d !important; }
    #MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; }
    .block-container { padding-top: 1.2rem !important; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "chain"        not in st.session_state: st.session_state.chain        = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "last_sources" not in st.session_state: st.session_state.last_sources = []
if "chain_loaded" not in st.session_state: st.session_state.chain_loaded = False
if "active_tab"   not in st.session_state: st.session_state.active_tab   = "chat"


# ─────────────────────────────────────────────
# LAZY LOAD CHAIN
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_chain():
    from chain import HealthcareQAChain
    return HealthcareQAChain()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MedFAQ RAG")
    st.markdown("---")

    st.markdown("**Navigation**")
    if st.button("💬  Chat Assistant", use_container_width=True):
        st.session_state.active_tab = "chat"
        st.rerun()
    if st.button("📊  Evaluation Dashboard", use_container_width=True):
        st.session_state.active_tab = "eval"
        st.rerun()

    st.markdown("---")
    st.markdown("**Session**")
    if st.button("🔄  Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_sources = []
        if st.session_state.chain:
            st.session_state.chain.reset_memory()
        st.rerun()

    st.markdown("---")
    st.markdown("**System Status**")
    if st.session_state.chain_loaded:
        st.success("✅ RAG System Ready")
    else:
        st.warning("⏳ Not initialized")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; line-height:1.9; color:#6e7681'>
        <b style='color:#8b949e'>Dataset</b><br>
        MedQuAD &middot; NIH (16.4k rows)<br><br>
        <b style='color:#8b949e'>LLM</b><br>
        Llama 3.3 70B via Groq<br><br>
        <b style='color:#8b949e'>Embeddings</b><br>
        all-MiniLM-L6-v2 (local)<br><br>
        <b style='color:#8b949e'>Vector DB</b><br>
        FAISS (local, CPU)<br><br>
        <b style='color:#8b949e'>Evaluation</b><br>
        RAGAS framework
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div style="font-size:2.2rem">🏥</div>
    <div>
        <h1>Healthcare FAQ Assistant</h1>
        <p>RAG &middot; NIH MedQuAD &middot; Llama 3.3 70B via Groq &middot; FAISS &middot; RAGAS Evaluation</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB: CHAT
# ─────────────────────────────────────────────
if st.session_state.active_tab == "chat":

    if not st.session_state.chain_loaded:
        with st.spinner("Loading RAG system..."):
            try:
                st.session_state.chain = load_chain()
                st.session_state.chain_loaded = True
            except Exception as e:
                st.error(f"Failed to load chain: {e}")
                st.stop()

    col_chat, col_sources = st.columns([2, 1], gap="large")

    with col_chat:
        st.markdown("### 💬 Ask a Medical Question")

        if not st.session_state.chat_history:
            st.markdown("""
            <div class='empty-state'>
                <div class='icon'>🩺</div>
                <p>Ask about symptoms, treatments, prevention, or diagnosis.<br>
                All answers are grounded in verified NIH sources.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for turn in st.session_state.chat_history:
                st.markdown("<div class='chat-label'>You</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chat-user'>{turn['question']}</div>", unsafe_allow_html=True)
                if turn.get("fallback"):
                    st.markdown(f"""
                    <div class='chat-label'>Assistant ⚠️</div>
                    <div class='fallback-box'>⚠️ <b>No verified context found.</b><br>{turn['answer']}</div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("<div class='chat-label'>Assistant</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='chat-assistant'>{turn['answer']}</div>", unsafe_allow_html=True)

        st.markdown("---")

        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Question",
                placeholder="e.g. What are the symptoms of Type 2 Diabetes?",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Ask →", use_container_width=True)

        st.markdown("<div style='font-size:0.78rem; color:#484f58; margin:8px 0 4px'>Try asking:</div>", unsafe_allow_html=True)
        suggestions = [
            "What are the symptoms of asthma?",
            "How is hypertension treated?",
            "How can I prevent Type 2 Diabetes?",
            "How is pneumonia diagnosed?",
        ]
        s_cols = st.columns(2)
        for i, sug in enumerate(suggestions):
            if s_cols[i % 2].button(sug, key=f"sug_{i}", use_container_width=True):
                with st.spinner("Searching medical knowledge base..."):
                    result = st.session_state.chain.ask(sug)
                st.session_state.chat_history.append({
                    "question": sug,
                    "answer":   result["answer"],
                    "fallback": result["fallback"],
                    "sources":  result["sources"],
                })
                st.session_state.last_sources = result["sources"]
                st.rerun()

        if submitted and user_input.strip():
            with st.spinner("Searching medical knowledge base..."):
                result = st.session_state.chain.ask(user_input.strip())
            st.session_state.chat_history.append({
                "question": user_input.strip(),
                "answer":   result["answer"],
                "fallback": result["fallback"],
                "sources":  result["sources"],
            })
            st.session_state.last_sources = result["sources"]
            st.rerun()

        st.markdown("""
        <div class='disclaimer'>
            ⚕️ <b>Medical Disclaimer:</b> For educational purposes only, using NIH MedQuAD data.
            Always consult a qualified healthcare professional for medical advice.
        </div>
        """, unsafe_allow_html=True)

    with col_sources:
        st.markdown("### 📚 Sources")
        sources = st.session_state.last_sources

        if not sources:
            st.markdown("""
            <div class='empty-state' style='padding:20px 0'>
                <div class='icon' style='font-size:1.8rem'>📄</div>
                <p>Retrieved NIH sources will appear here after each answer.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='font-size:0.78rem; color:#484f58; margin-bottom:10px'>{len(sources)} chunk(s) from NIH MedQuAD</div>", unsafe_allow_html=True)
            for src in sources:
                qtype    = src.get("qtype", "general")
                question = src.get("original_question", "N/A")
                short_q  = question[:110] + ("..." if len(question) > 110 else "")
                st.markdown(f"""
                <div class='source-card'>
                    <span class='source-qtype'>{qtype}</span><br>
                    <b>Source Q:</b> {short_q}
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<div style='font-size:0.73rem; color:#484f58; margin-top:6px'>Verified NIH guidelines via MedQuAD.</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB: EVALUATION DASHBOARD
# ─────────────────────────────────────────────
elif st.session_state.active_tab == "eval":

    st.markdown("### 📊 RAGAS Evaluation Dashboard")
    st.markdown("<div style='color:#6e7681; font-size:0.88rem; margin-bottom:16px'>A/B test: RAG System vs Baseline LLM &middot; Judge: Llama 3.3 70B via Groq &middot; Embedder: all-MiniLM-L6-v2 (local)</div>", unsafe_allow_html=True)
    st.markdown("---")

    RESULTS_DIR = "data/eval_results"
    existing_results = []
    if os.path.exists(RESULTS_DIR):
        existing_results = sorted(
            [f for f in os.listdir(RESULTS_DIR) if f.startswith("scores_") and f.endswith(".json")],
            reverse=True,
        )

    col_run, col_history = st.columns([1, 1], gap="large")

    with col_run:
        st.markdown("#### ▶️ Run New Evaluation")
        sample_size = st.slider(
            "Number of questions",
            min_value=10, max_value=100, value=10, step=10,
            help="Each question makes 2 Groq LLM calls. Start with 10 to verify everything works."
        )
        qtypes_available = ["symptoms", "treatment", "prevention", "causes", "exams and tests"]
        selected_qtypes  = st.multiselect(
            "Question types to include",
            options=qtypes_available,
            default=qtypes_available,
        )
        run_button = st.button("▶️  Run Evaluation", use_container_width=True)

        if run_button:
            if not selected_qtypes:
                st.error("Select at least one question type.")
            else:
                if not st.session_state.chain_loaded:
                    with st.spinner("Loading chain..."):
                        st.session_state.chain        = load_chain()
                        st.session_state.chain_loaded = True

                progress_bar = st.progress(0)
                status_text  = st.empty()
                try:
                    from eval import (
                        build_eval_dataset, run_rag_system,
                        run_baseline_llm, score_with_ragas, save_results,
                    )
                    import eval as eval_module

                    eval_module.EVAL_SAMPLE_SIZE  = sample_size
                    eval_module.EVAL_QTYPES       = selected_qtypes
                    eval_module.SAMPLES_PER_QTYPE = max(1, sample_size // len(selected_qtypes))

                    status_text.text("Step 1/4 — Building evaluation dataset...")
                    eval_df = build_eval_dataset()
                    progress_bar.progress(25)

                    status_text.text("Step 2/4 — Running RAG system...")
                    eval_df = run_rag_system(eval_df)
                    progress_bar.progress(50)

                    status_text.text("Step 3/4 — Running baseline LLM...")
                    eval_df = run_baseline_llm(eval_df)
                    progress_bar.progress(75)

                    status_text.text("Step 4/4 — Scoring with RAGAS...")
                    rag_scores      = score_with_ragas(eval_df, answer_col="rag_answer")
                    baseline_scores = score_with_ragas(eval_df, answer_col="baseline_answer")
                    save_results(eval_df, rag_scores, baseline_scores)
                    progress_bar.progress(100)
                    status_text.success("✅ Evaluation complete!")

                    st.session_state["eval_rag"]     = rag_scores
                    st.session_state["eval_baseline"] = baseline_scores
                    st.session_state["eval_df"]       = eval_df
                    st.rerun()
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

    with col_history:
        st.markdown("#### 🕑 Past Runs")
        if not existing_results:
            st.info("No past evaluations yet. Run one to see results here.")
        else:
            selected_run = st.selectbox(
                "Load a past run",
                options=existing_results,
                format_func=lambda x: x.replace("scores_", "").replace(".json", ""),
            )
            if selected_run:
                with open(os.path.join(RESULTS_DIR, selected_run)) as f:
                    past = json.load(f)
                st.session_state["eval_rag"]     = past["rag_scores"]
                st.session_state["eval_baseline"] = past["baseline_scores"]
                st.success("Loaded: " + selected_run.replace("scores_", "").replace(".json", ""))

    if "eval_rag" in st.session_state:
        st.markdown("---")
        rag_s  = st.session_state["eval_rag"]
        base_s = st.session_state["eval_baseline"]

        metrics = [
            ("Faithfulness",       "faithfulness",       "Grounded in retrieved context?"),
            ("Answer Correctness", "answer_correctness", "Match to NIH ground truth"),
            ("Context Precision",  "context_precision",  "Retrieval signal-to-noise ratio"),
        ]

        st.markdown("**🤖 RAG System**")
        rag_cols = st.columns(3)
        for i, (label, key, desc) in enumerate(metrics):
            val      = rag_s[key]
            base_val = base_s[key]

            if isinstance(val, float) and isinstance(base_val, float):
                delta      = val - base_val
                arrow      = "↑" if delta >= 0 else "↓"
                cls        = "metric-delta-pos" if delta >= 0 else "metric-delta-neg"
                delta_html = f"<div class='{cls}'>{arrow} {abs(delta):.3f} vs baseline</div>"
                val_display = f"{val:.3f}"
            else:
                delta_html  = "<div style='color:#484f58; font-size:0.8rem'>not evaluated</div>"
                val_display = str(val)

            rag_cols[i].markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val_display}</div>
                <div class='metric-label'>{label}</div>
                {delta_html}
                <div style='font-size:0.69rem; color:#484f58; margin-top:5px'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("**📉 Baseline LLM (no retrieval)**")
        base_cols = st.columns(3)
        for i, (label, key, _) in enumerate(metrics):
            val         = base_s[key]
            val_display = f"{val:.3f}" if isinstance(val, float) else str(val)
            base_cols[i].markdown(f"""
            <div class='metric-card' style='border-top: 2px solid #30363d'>
                <div class='metric-baseline'>{val_display}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

        if "eval_df" in st.session_state:
            st.markdown("---")
            st.markdown("#### 📋 Detailed Results")
            display_df = st.session_state["eval_df"][[
                "question", "qtype", "ground_truth",
                "rag_answer", "baseline_answer", "rag_fallback"
            ]].copy()
            display_df.columns = [
                "Question", "QType", "Ground Truth (NIH)",
                "RAG Answer", "Baseline Answer", "Fallback?"
            ]
            st.dataframe(display_df, use_container_width=True, height=380)
            csv = display_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️  Download Results CSV",
                data=csv,
                file_name=f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )