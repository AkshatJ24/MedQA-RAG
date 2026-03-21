"""
ingestion.py — MedQuAD RAG Ingestion Pipeline
Dataset: keivalya/MedQuad-MedicalQnADataset (HuggingFace)
Columns: qtype | Question | Answer
"""

import os
import logging
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_NAME    = "keivalya/MedQuad-MedicalQnADataset"
FAISS_INDEX_PATH = "data/faiss_index"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

CHUNK_SIZE    = 700
CHUNK_OVERLAP = 100
MIN_CHUNK_LEN = 60    # discard near-empty chunks
MAX_CHUNK_LEN = 1500  # discard runaway chunks

# question types where the FULL answer should stay as one chunk
# (short, structured answers — splitting breaks their meaning)
NO_SPLIT_QTYPES = {"treatment", "prevention", "exams and tests"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# STEP 1: Load dataset
# ─────────────────────────────────────────────
def load_medquad():
    log.info(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")
    log.info(f"Loaded {len(dataset)} rows.")
    return dataset


# ─────────────────────────────────────────────
# STEP 2: Convert rows → LangChain Documents
# Each row becomes a Document where the content
# is a combined Q+A block (improves retrieval context)
# ─────────────────────────────────────────────
def build_documents(dataset):
    docs = []
    skipped = 0

    for idx, row in enumerate(dataset):
        qtype    = (row.get("qtype") or "general").strip().lower()
        question = (row.get("Question") or "").strip()
        answer   = (row.get("Answer") or "").strip()

        # Skip rows with missing answer
        if not answer or len(answer) < MIN_CHUNK_LEN:
            skipped += 1
            continue

        # Combine Q+A so the retriever gets full context
        content = f"Question: {question}\nAnswer: {answer}"

        docs.append(Document(
            page_content=content,
            metadata={
                "qtype":        qtype,       # e.g. "symptoms", "treatment"
                "question":     question,    # original question (useful for RAGAS)
                "source":       DATASET_NAME,
                "row_index":    idx,
                "answer_len":   len(answer),
            }
        ))

    log.info(f"Built {len(docs)} documents. Skipped {skipped} empty rows.")
    return docs


# ─────────────────────────────────────────────
# STEP 3: Smart chunking
# — Short/structured qtypes: keep whole (no split)
# — Long answers (symptoms, susceptibility): chunk
# ─────────────────────────────────────────────
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )

    chunked = []
    for doc in docs:
        qtype = doc.metadata.get("qtype", "")

        # For structured short-answer types, keep the document whole
        if qtype in NO_SPLIT_QTYPES or len(doc.page_content) <= CHUNK_SIZE:
            chunked.append(doc)
        else:
            splits = splitter.split_documents([doc])
            # Propagate all metadata to every child chunk
            for i, chunk in enumerate(splits):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["total_chunks"] = len(splits)
                chunked.append(chunk)

    log.info(f"Total chunks after splitting: {len(chunked)}")
    return chunked


# ─────────────────────────────────────────────
# STEP 4: Validate — remove noise chunks
# ─────────────────────────────────────────────
NOISE_PREFIXES = (
    "click here", "see also", "references", "for more information",
    "last reviewed", "source:", "related topics",
)

def validate_chunks(chunks):
    valid = []
    for chunk in chunks:
        text = chunk.page_content.strip()
        length = len(text)

        if length < MIN_CHUNK_LEN or length > MAX_CHUNK_LEN:
            continue
        if any(text.lower().startswith(p) for p in NOISE_PREFIXES):
            continue

        valid.append(chunk)

    removed = len(chunks) - len(valid)
    log.info(f"Validation: kept {len(valid)} chunks, removed {removed} noisy chunks.")
    return valid


# ─────────────────────────────────────────────
# STEP 5: Embed + Index in FAISS
# ─────────────────────────────────────────────
def build_vectorstore(chunks):
    log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={
            "batch_size": 64,
            "normalize_embeddings": True,   # required for cosine similarity in FAISS
        }
    )

    log.info("Building FAISS index — this may take a few minutes...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, embeddings


# ─────────────────────────────────────────────
# STEP 6: Save / Load (idempotent)
# ─────────────────────────────────────────────
def get_vectorstore(force_rebuild=False):
    """
    Returns a FAISS vectorstore.
    Loads from disk if already built; rebuilds if not or force_rebuild=True.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"batch_size": 64, "normalize_embeddings": True}
    )

    if not force_rebuild and os.path.exists(FAISS_INDEX_PATH):
        log.info(f"Loading existing FAISS index from: {FAISS_INDEX_PATH}")
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # Full rebuild
    dataset    = load_medquad()
    docs       = build_documents(dataset)
    chunks     = chunk_documents(docs)
    chunks     = validate_chunks(chunks)
    vectorstore, _ = build_vectorstore(chunks)

    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)
    log.info(f"FAISS index saved to: {FAISS_INDEX_PATH}")

    return vectorstore


# ─────────────────────────────────────────────
# STEP 7: Filtered retriever (bonus utility)
# Use in chain.py to restrict search by qtype
# ─────────────────────────────────────────────
def get_filtered_retriever(vectorstore, qtype_filter: str = None, k: int = 5):
    """
    Returns a retriever. If qtype_filter is provided, only returns
    chunks of that question type (e.g. 'symptoms', 'treatment').

    Usage in chain.py:
        retriever = get_filtered_retriever(vectorstore, qtype_filter="treatment")
    """
    search_kwargs = {"k": k}

    if qtype_filter:
        search_kwargs["filter"] = {"qtype": qtype_filter}

    return vectorstore.as_retriever(search_kwargs=search_kwargs)


# ─────────────────────────────────────────────
# Entry point — run directly to build index
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MedQuAD Ingestion Pipeline")
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force rebuild the FAISS index even if it exists"
    )
    args = parser.parse_args()

    vectorstore = get_vectorstore(force_rebuild=args.rebuild)

    # Quick sanity check
    test_query = "What are the symptoms of diabetes?"
    results = vectorstore.similarity_search_with_score(test_query, k=3)

    print("\n--- Sanity Check ---")
    for doc, score in results:
        print(f"[score: {score:.4f}] [{doc.metadata.get('qtype')}] {doc.page_content[:120]}...")

    print("\n✅ Ingestion complete.")