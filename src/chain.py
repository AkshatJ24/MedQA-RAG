"""
chain.py — RAG Chain for Healthcare FAQ Assistant
Wires up: FAISS vectorstore → Groq LLM → LCEL chain
Features:
  - Auto qtype detection to filter retrieval
  - Similarity score threshold (zero-guessing policy)
  - Source attribution
  - Conversation memory (multi-turn)
"""

import os
import logging
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from ingestion import get_vectorstore, get_filtered_retriever

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GROQ_API_KEY        = os.getenv("GROQ_API_KEY")
SIMILARITY_THRESHOLD = 0.70   # below this score → trigger fallback
TOP_K               = 5       # number of chunks to retrieve

FALLBACK_RESPONSE = (
    "I do not have enough verified information to answer this question. "
    "Please consult a qualified healthcare professional for accurate medical advice."
)

# Maps keywords in the user query → MedQuAD qtype
# Used to auto-filter the retriever for better precision
QTYPE_KEYWORD_MAP = {
    "symptoms":         ["symptom", "signs", "feel", "experience", "suffer"],
    "treatment":        ["treat", "treatment", "therapy", "medicine", "medication", "cure", "manage"],
    "prevention":       ["prevent", "prevention", "avoid", "reduce risk", "protect"],
    "causes":           ["cause", "why", "reason", "leads to", "results in"],
    "susceptibility":   ["risk", "at risk", "prone", "vulnerable", "who gets"],
    "exams and tests":  ["diagnose", "diagnosis", "test", "exam", "detect", "check"],
    "inheritance":      ["genetic", "inherit", "hereditary", "family history"],
    "outlook":          ["prognosis", "outlook", "survival", "life expectancy", "recover"],
}


# ─────────────────────────────────────────────
# SYSTEM PROMPT  (modern ChatPromptTemplate)
# Strict grounding — LLM must only use context
# ─────────────────────────────────────────────
SYSTEM_MESSAGE = """You are a Healthcare FAQ Assistant. Your role is to answer medical questions
accurately and safely using ONLY the context provided below.

STRICT RULES:
1. Answer ONLY using information from the context. Do NOT use outside knowledge.
2. If the context does not contain enough information, say exactly:
   "I do not have enough verified information to answer this. Please consult a healthcare professional."
3. Never guess, speculate, or fabricate medical information.
4. Keep your answer clear, concise, and easy to understand for a general audience.
5. If the context contains a list (e.g. symptoms or treatment steps), preserve that structure.

Context:
{context}"""

# ChatPromptTemplate supports multi-turn history natively via MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MESSAGE),
    MessagesPlaceholder(variable_name="chat_history"),   # ← multi-turn memory slot
    ("human", "{input}"),
])


# ─────────────────────────────────────────────
# QTYPE DETECTION
# Detects the likely qtype from a user query
# so we can narrow the FAISS search
# ─────────────────────────────────────────────
def detect_qtype(query: str) -> str | None:
    """
    Returns a qtype string if a keyword match is found, else None.

    Example:
        detect_qtype("How do I treat diabetes?") → "treatment"
        detect_qtype("What causes asthma?")       → "causes"
        detect_qtype("Tell me about lupus")        → None  (no filter)
    """
    query_lower = query.lower()
    for qtype, keywords in QTYPE_KEYWORD_MAP.items():
        if any(kw in query_lower for kw in keywords):
            log.info(f"Detected qtype: '{qtype}' for query.")
            return qtype
    log.info("No specific qtype detected — using unfiltered retrieval.")
    return None


# ─────────────────────────────────────────────
# SIMILARITY GATE
# Checks if retrieved docs are relevant enough
# before passing them to Gemini
# ─────────────────────────────────────────────
def is_context_reliable(vectorstore, query: str) -> tuple[bool, list]:
    """
    Runs a similarity search with scores.
    Returns (True, docs) if top result clears the threshold.
    Returns (False, []) if context is too weak → fallback should trigger.

    Note on FAISS scores:
        FAISS returns L2 distance by default (lower = more similar).
        With normalize_embeddings=True in ingestion.py, the range is [0, 2].
        We convert to cosine similarity: cos_sim = 1 - (score / 2)
        and check against SIMILARITY_THRESHOLD.
    """
    results = vectorstore.similarity_search_with_score(query, k=TOP_K)

    if not results:
        return False, []

    top_score = results[0][1]  # L2 distance of best match
    cos_sim   = 1 - (top_score / 2)

    log.info(f"Top similarity score: {cos_sim:.4f} (threshold: {SIMILARITY_THRESHOLD})")

    if cos_sim < SIMILARITY_THRESHOLD:
        log.warning("Context below similarity threshold — triggering fallback.")
        return False, []

    docs = [doc for doc, _ in results]
    return True, docs


# ─────────────────────────────────────────────
# LLM SETUP
# ─────────────────────────────────────────────
def get_llm():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")

    return ChatOpenAI(
        model="llama-3.3-70b-versatile",  # best free Groq model
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.2,
        max_tokens=512,
    )


# ─────────────────────────────────────────────
# MAIN CHAIN CLASS
# ─────────────────────────────────────────────
class HealthcareQAChain:
    def __init__(self):
        log.info("Initializing HealthcareQAChain...")
        self.vectorstore = get_vectorstore()
        self.llm         = get_llm()
        # Modern memory: plain list of HumanMessage / AIMessage objects
        # Replaces the deprecated ConversationBufferMemory
        self.chat_history: list = []
        log.info("Chain ready.")

    def ask(self, query: str) -> dict:
        """
        Main entry point. Pass a user question, get back:
        {
            "answer":   str,
            "sources":  list[dict],   # [{question, qtype, source}]
            "fallback": bool          # True if zero-guessing policy triggered
        }
        """
        # 1. Check if context is reliable enough
        reliable, docs = is_context_reliable(self.vectorstore, query)

        if not reliable:
            return {
                "answer":   FALLBACK_RESPONSE,
                "sources":  [],
                "fallback": True,
            }

        # 2. Detect qtype and get appropriate retriever
        qtype     = detect_qtype(query)
        retriever = get_filtered_retriever(
            self.vectorstore,
            qtype_filter=qtype,
            k=TOP_K
        )

        # 3. Build pure LCEL chain using only langchain_core
        #
        #  How the pipe works step by step:
        #
        #  Input dict
        #    │
        #    ├─ "context"      ← retriever fetches docs → format_docs joins them
        #    ├─ "input"        ← user query passed through unchanged
        #    └─ "chat_history" ← message list passed through unchanged
        #    │
        #    ▼
        #  prompt   (ChatPromptTemplate fills system + history + human slots)
        #    ▼
        #  self.llm (Gemini generates the answer)
        #    ▼
        #  StrOutputParser (extracts plain string from AIMessage)

        def format_docs(docs):
            """Joins retrieved chunk texts into one context string."""
            return "\n\n".join(doc.page_content for doc in docs)

        # We need docs for both context text AND source attribution later,
        # so retrieve once and store in a shared variable
        retrieved_docs = retriever.invoke(query)

        rag_chain = (
            {
                "context":      RunnableLambda(lambda _: format_docs(retrieved_docs)),
                "input":        RunnablePassthrough(),
                "chat_history": RunnableLambda(lambda _: self.chat_history),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # 4. Run
        answer = rag_chain.invoke(query)
        if not answer:
            answer = FALLBACK_RESPONSE

        # 5. Update memory with this turn
        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=answer))

        # 6. Extract source attribution from already-retrieved docs
        source_docs = retrieved_docs
        sources = [
            {
                "original_question": doc.metadata.get("question", "N/A"),
                "qtype":             doc.metadata.get("qtype", "N/A"),
                "source":            doc.metadata.get("source", "N/A"),
            }
            for doc in source_docs
        ]

        return {
            "answer":   answer,
            "sources":  sources,
            "fallback": False,
        }

    def reset_memory(self):
        """Clears conversation history. Call this to start a fresh session."""
        self.chat_history = []
        log.info("Conversation memory cleared.")


# ─────────────────────────────────────────────
# Quick CLI test — run directly to try the chain
# ─────────────────────────────────────────────
if __name__ == "__main__":
    chain = HealthcareQAChain()

    test_queries = [
        "What are the symptoms of diabetes?",
        "How is diabetes treated?",
        "What are the side effects of paracetamol overdose?",
        "What is the best stock to invest in today?",  # out-of-domain → fallback
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Q: {query}")
        result = chain.ask(query)
        print(f"A: {result['answer']}")
        if result["fallback"]:
            print("⚠️  [Fallback triggered — no reliable context found]")
        else:
            print(f"\n📚 Sources ({len(result['sources'])}):")
            for s in result["sources"]:
                print(f"  - [{s['qtype']}] {s['original_question'][:80]}")