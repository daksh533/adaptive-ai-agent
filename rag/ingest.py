"""
rag/ingest.py

Phase 5 — RAG Pipeline

Loads documents from a local knowledge base, chunks them with overlap,
embeds each chunk using a sentence-transformers model, and stores them
in a persistent ChromaDB collection so the agent can retrieve them later.

Usage:
    python -m rag.ingest              # ingest all docs in rag/knowledge_base/
    python -m rag.ingest --reset      # wipe collection first, then ingest
"""

import os
import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE_DIR = Path(__file__).parent / "knowledge_base"
CHROMA_DIR = Path(__file__).parent / "chroma_store"
COLLECTION_NAME = "agent_knowledge"

# Embedding model (local, no API key needed)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 400        # characters per chunk (approximates ~300-500 tokens)
CHUNK_OVERLAP = 80      # characters of overlap between consecutive chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split *text* into overlapping chunks of approximately *chunk_size* characters.

    Args:
        text:       The full document text.
        chunk_size: Target size of each chunk in characters.
        overlap:    Number of characters to repeat at the start of the next chunk.

    Returns:
        A list of non-empty chunk strings.
    """
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Advance by chunk_size minus overlap so consecutive chunks share context
        start += chunk_size - overlap

    return chunks


def load_document(path: Path) -> str:
    """
    Read a plain-text, markdown, or .txt file and return its content as a string.
    Falls back to latin-1 if UTF-8 decoding fails.
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


# ---------------------------------------------------------------------------
# Main ingestion logic
# ---------------------------------------------------------------------------

def ingest(reset: bool = False) -> None:
    """
    Walk KNOWLEDGE_BASE_DIR, chunk every supported document, embed each chunk,
    and upsert them into the persistent ChromaDB collection.

    Args:
        reset: If True, delete the existing collection before ingesting.
    """
    # 1. Initialise persistent Chroma client
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"[Ingest] Deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass  # Collection didn't exist yet

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # use cosine similarity
    )

    # 2. Load embedding model
    print(f"[Ingest] Loading embedding model '{EMBED_MODEL_NAME}'...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # 3. Walk knowledge base directory
    if not KNOWLEDGE_BASE_DIR.exists():
        raise FileNotFoundError(
            f"Knowledge base directory not found: {KNOWLEDGE_BASE_DIR}\n"
            "Create rag/knowledge_base/ and add .txt / .md documents to it."
        )

    supported_extensions = {".txt", ".md"}
    doc_paths = [
        p for p in KNOWLEDGE_BASE_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in supported_extensions
    ]

    if not doc_paths:
        print("[Ingest] No supported documents found in knowledge_base/. Nothing to ingest.")
        return

    total_chunks = 0

    for doc_path in sorted(doc_paths):
        relative_name = doc_path.relative_to(KNOWLEDGE_BASE_DIR)
        print(f"\n[Ingest] Processing: {relative_name}")

        text = load_document(doc_path)
        chunks = chunk_text(text)

        print(f"           {len(chunks)} chunks extracted from {len(text)} characters")

        # Build unique IDs: <filename>_chunk_<index>
        doc_stem = str(relative_name).replace("\\", "/").replace(" ", "_")
        ids = [f"{doc_stem}_chunk_{i}" for i in range(len(chunks))]

        # Embed all chunks for this document in one batch
        embeddings = model.encode(chunks, show_progress_bar=False).tolist()

        # Metadata attached to every chunk for citation
        metadatas = [
            {
                "source": str(relative_name).replace("\\", "/"),
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            for i in range(len(chunks))
        ]

        # Upsert (safe to re-run: duplicate IDs are updated, not duplicated)
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        total_chunks += len(chunks)
        print(f"           Upserted {len(chunks)} chunks into ChromaDB.")

    print(f"\n[Ingest] Done! {len(doc_paths)} document(s) -> {total_chunks} total chunks stored.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG knowledge base.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing ChromaDB collection before ingesting.",
    )
    args = parser.parse_args()
    ingest(reset=args.reset)

