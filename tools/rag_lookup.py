"""
tools/rag_lookup.py

Phase 5 — RAG Pipeline

Wraps the RAG retriever as a standard agent tool so the LLM executor
can call it exactly like calculate, web_search, or run_python.

When the LLM needs to ground an answer in the knowledge base it emits:

    {
        "name": "rag_lookup",
        "arguments": {
            "query": "How does self-correction work in AI agents?",
            "top_k": 4
        }
    }

The tool returns a formatted string containing the most relevant chunks
and their source citations, which the LLM then uses to compose its answer.
"""

from rag.retriever import retrieve


# ---------------------------------------------------------------------------
# Tool schema (OpenAI function-calling format)
# ---------------------------------------------------------------------------

RAG_LOOKUP_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "rag_lookup",
        "description": (
            "Search the local knowledge base and return the most relevant "
            "document chunks for a given query. "
            "Use this tool when you need to ground your answer in specific "
            "documents or when the question can be answered from the "
            "pre-loaded knowledge base. "
            "Always cite the source document and chunk index in your final answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The natural-language search query to look up in "
                        "the knowledge base."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": (
                        "Number of document chunks to retrieve. "
                        "Defaults to 4. Maximum is 10."
                    ),
                    "default": 4,
                },
            },
            "required": ["query"],
        },
    },
}


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handle_tool_call(arguments: dict) -> str:
    """
    Execute a rag_lookup tool call.

    Args:
        arguments: Dict with keys:
                     - "query"  (str, required)
                     - "top_k"  (int, optional, default 4)

    Returns:
        A formatted string containing retrieved chunks with source citations,
        or an error message if retrieval fails.
    """
    query = arguments.get("query", "").strip()
    top_k = int(arguments.get("top_k", 4))

    if not query:
        return "Error: 'query' argument is required for rag_lookup."

    try:
        chunks = retrieve(query, top_k=top_k)
    except RuntimeError as exc:
        return (
            f"RAG knowledge base is not available: {exc}\n"
            "Ask the user to run 'python -m rag.ingest' first."
        )
    except Exception as exc:
        return f"RAG lookup failed: {exc}"

    if not chunks:
        return f"No relevant chunks found in the knowledge base for query: '{query}'"

    # Format results for the LLM
    lines = [f"RAG results for query: '{query}'\n"]
    for i, chunk in enumerate(chunks, start=1):
        lines.append(
            f"--- Result {i} ---\n"
            f"Source : {chunk['source']} (chunk {chunk['chunk_index']} "
            f"of {chunk['total_chunks']})\n"
            f"Similarity distance: {chunk['distance']}\n"
            f"Content:\n{chunk['text']}\n"
        )

    return "\n".join(lines)

