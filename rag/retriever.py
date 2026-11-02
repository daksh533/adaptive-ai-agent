"""
rag/retriever.py

Phase 5 — RAG Pipeline

Given a natural-language query, embeds it with the same sentence-transformers
model used during ingestion, then asks ChromaDB to return the top-k most
similar chunks together with their source metadata.

Usage (programmatic):
    from rag.retriever import retrieve
    results = retrieve("What is self-correction in AI agents?", top_k=3)
    for r in results:
        print(r["source"], r["chunk_index"], r["text"])
"""

import os
from pathlib import Path

# Prevent unnecessary network pings when model is already cached locally
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from sentence_transformers import SentenceTransformer
import chromadb


# ---------------------------------------------------------------------------
# Configuration  (must match rag/ingest.py)
# ---------------------------------------------------------------------------

CHROMA_DIR = Path(__file__).parent / "chroma_store"
COLLECTION_NAME = "agent_knowledge"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_TOP_K = 4          # number of chunks returned by default
MAX_TOP_K = 10             # hard ceiling to avoid returning the entire corpus


# ---------------------------------------------------------------------------
# Module-level singletons (loaded once, reused across calls)
# ---------------------------------------------------------------------------

_model: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        if not CHROMA_DIR.exists():
            raise RuntimeError(
                "ChromaDB store not found. Run 'python -m rag.ingest' first."
            )
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Find the top-k most relevant document chunks for *query*.

    Args:
        query:  The search query (natural language).
        top_k:  Number of chunks to return (capped at MAX_TOP_K).

    Returns:
        A list of dicts, each containing:
            - "text":         the chunk text
            - "source":       filename relative to knowledge_base/
            - "chunk_index":  integer position of the chunk within that document
            - "total_chunks": total number of chunks in that document
            - "distance":     cosine distance (lower = more similar)

    Raises:
        RuntimeError: if the ChromaDB store has not been created yet.
    """
    top_k = min(top_k, MAX_TOP_K)

    model = _get_model()
    collection = _get_collection()

    # Embed the query
    query_embedding = model.encode(query, show_progress_bar=False).tolist()

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Flatten the nested lists Chroma returns (one query → one outer list)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    output = []
    for text, meta, dist in zip(docs, metas, distances):
        output.append(
            {
                "text": text,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "total_chunks": meta.get("total_chunks", -1),
                "distance": round(float(dist), 4),
            }
        )

    return output


def retrieve_as_context(query: str, top_k: int = DEFAULT_TOP_K) -> str:
    """
    Convenience wrapper that formats the retrieved chunks as a single
    context string ready to be injected into an LLM prompt.

    Args:
        query:  The search query.
        top_k:  Number of chunks to retrieve.

    Returns:
        A formatted multi-line string with each chunk labelled by its source.
    """
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return "No relevant documents found in the knowledge base."

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i}: {chunk['source']} | chunk {chunk['chunk_index']}]\n"
            f"{chunk['text']}"
        )

    return "\n\n---\n\n".join(parts)

