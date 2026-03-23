---
title: MedQA v2.0
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.43.0
app_file: app.py
pinned: false
license: mit
short_description: RAG-powered medical Q&A grounded in NIH MedQuAD sources
---

# 🏥 Healthcare FAQ Assistant

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/iamAkshat/MedQA-RAG-assistant)
[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**🚀 Live Demo:** [huggingface.co/spaces/iamAkshat/MedQA-RAG-assistant](https://huggingface.co/spaces/iamAkshat/MedQA-RAG-assistant)

---

A production-ready **Retrieval-Augmented Generation (RAG)** system for medical question answering, grounded in verified NIH sources. Built to address the critical problem of LLM hallucinations in healthcare — where incorrect information can cause real harm.

---

## 📊 Evaluation Results

Evaluated on **50 questions** across 5 medical categories using RAGAS (LLM-as-judge):

| Metric | RAG System | Baseline LLM | Delta |
|---|---|---|---|
| **Faithfulness** | **0.4937** | 0.3916 | ↑ +0.1021 (+26%) |

> **Faithfulness** measures whether every statement in the generated answer is grounded in the retrieved NIH context — directly measuring hallucination reduction. A RAG system scoring 26% higher than a baseline LLM with no retrieval demonstrates that grounding in verified sources measurably improves answer reliability.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    SAFETY GATE                              │
│  FAISS similarity search → score < 0.70 → Fallback         │
│                           → score ≥ 0.70 → Continue        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  RAG PIPELINE                               │
│                                                             │
│  1. Detect qtype from query keywords                        │
│     (symptoms / treatment / prevention / ...)              │
│                                                             │
│  2. FAISS filtered retrieval                                │
│     → Top-5 chunks from MedQuAD index                      │
│                                                             │
│  3. Llama 3.3 70B via Groq                                  │
│     → Strict system prompt: answer ONLY from context       │
│                                                             │
│  4. Source attribution                                      │
│     → Returns NIH source metadata with every answer        │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

**Safety-First Design**
- Zero-Guessing Policy — if FAISS similarity score < 0.70, the system returns a hard fallback instead of hallucinating
- Strict contextual grounding — system prompt forces the LLM to answer only from retrieved NIH context
- Source attribution — every answer includes the original NIH MedQuAD source so users can verify

**Smart Retrieval**
- Auto qtype detection — detects whether a query is about symptoms, treatment, prevention, etc. and narrows FAISS search accordingly
- Semantic chunking strategy — structured answers (treatment, prevention) are kept whole to avoid breaking clinical meaning; long answers (symptoms) are split for coverage
- Normalized embeddings — cosine similarity via FAISS for accurate relevance scoring

**Quantitative Evaluation**
- RAGAS faithfulness metric with LLM-as-judge
- A/B comparison against a baseline LLM (no retrieval)
- Per-qtype breakdown across 5 medical question categories
- Results saved as timestamped CSV + JSON

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| Orchestration | LangChain (LCEL) |
| LLM | Llama 3.3 70B via Groq API |
| Embeddings | all-MiniLM-L6-v2 (local, HuggingFace) |
| Vector Database | FAISS (local, CPU) |
| Dataset | MedQuAD — NIH Medical Q&A (16,400 rows) |
| Evaluation | RAGAS framework |
| UI | Streamlit (dark mode) |

---

## 📂 Project Structure

```
healthcare-rag-assistant/
├── src/
│   ├── ingestion.py      # Dataset loading, chunking, embedding, FAISS indexing
│   ├── chain.py          # RAG pipeline — retrieval + LLM + safety layer
│   ├── eval.py           # RAGAS A/B evaluation framework
│   └── app.py            # Streamlit dark mode UI
├── data/
│   ├── faiss_index/      # Auto-generated FAISS vector index (built on first run)
│   └── eval_results/     # Timestamped CSV + JSON evaluation results
├── .env                  # API keys (not committed to git)
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com) (no credit card required)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/AkshatJ24/MedQA-RAG.git
cd healthcare-rag-assistant

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
# Create a .env file in the root folder:
echo "GROQ_API_KEY=your_groq_key_here" > .env
```

### Running the Project

```bash
cd src

# Step 1 — Build the FAISS index (run once, takes ~3 minutes)
python ingestion.py

# Step 2 — Test the RAG pipeline
python chain.py

# Step 3 — Launch the Streamlit app
streamlit run app.py

# Step 4 — Run RAGAS evaluation (optional)
python eval.py --sample-size 10   # quick test
python eval.py --sample-size 50   # full evaluation
```

### Streamlit Configuration (Dark Mode)

Create `.streamlit/config.toml` in the project root:
```toml
[theme]
base = "dark"
```

---

## 📦 Requirements

```
langchain
langchain-core
langchain-community
langchain-huggingface
langchain-openai
langchain-text-splitters
faiss-cpu
sentence-transformers
datasets
pandas
python-dotenv
streamlit
ragas
openai
google-generativeai
```

---

## 📊 Dataset

**MedQuAD** (Medical Question Answering Dataset)
- Source: [keivalya/MedQuad-MedicalQnADataset](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset) on HuggingFace
- Origin: Verified NIH (National Institutes of Health) guidelines
- Size: 16,400 Q&A pairs
- Question types: symptoms, treatment, prevention, causes, susceptibility, exams and tests, inheritance, outlook

---

## 🔒 Safety Features

| Feature | Implementation |
|---|---|
| Similarity threshold | Queries scoring < 0.70 cosine similarity trigger hard fallback |
| Contextual grounding | System prompt prohibits the LLM from using outside knowledge |
| Zero-guessing policy | Hard-coded fallback: *"Please consult a healthcare professional"* |
| Source attribution | Every answer returns the originating NIH MedQuAD source |
| Out-of-domain rejection | Non-medical queries (e.g. finance, general knowledge) are refused |

---

## 📈 Evaluation Methodology

The RAGAS **faithfulness** metric works as follows:

1. The judge LLM decomposes the generated answer into individual statements
2. For each statement, it checks whether the statement can be inferred from the retrieved context
3. Faithfulness = (statements supported by context) / (total statements)

A score of 1.0 means fully grounded, 0.0 means completely hallucinated.

**Why faithfulness matters for healthcare:** In medical Q&A, an answer that sounds confident but contradicts the source is potentially dangerous. Faithfulness directly quantifies how often the system stays within verified NIH information vs. generating from model memory.

---

## ⚠️ Medical Disclaimer

This application is for **educational purposes only**. It provides information from the NIH MedQuAD dataset and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical decisions.

---

## 👤 Author

Built as a college-level RAG systems project demonstrating:
- Production-grade Python architecture
- Responsible AI design for sensitive domains
- Quantitative evaluation using industry-standard frameworks
