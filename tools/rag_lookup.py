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

from tools.knowledge_lookup import lookup_with_fallback


# ---------------------------------------------------------------------------
# Tool schema (OpenAI function-calling format)
# ---------------------------------------------------------------------------

RAG_LOOKUP_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "rag_lookup",
        "description": (
            "Search the local knowledge base first for a given query, and "
            "automatically fall back to live web search if the information is "
            "not found in the knowledge base. Always cite the source document "
            "and chunk index or web URL in your final answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The natural-language search query to look up in "
                        "the knowledge base or web."
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

    Checks the local RAG knowledge base first. If no relevant chunks are found
    or relevance distance is high, automatically falls back to live web search
    so the agent gets the information and advances to the next step immediately.

    Args:
        arguments: Dict with keys:
                     - "query"  (str, required)
                     - "top_k"  (int, optional, default 4)

    Returns:
        Source-labeled result string (from KB or web fallback).
    """
    query = arguments.get("query", "").strip()

    try:
        top_k = int(arguments.get("top_k", 4))
    except (TypeError, ValueError):
        top_k = 4
    top_k = max(1, top_k)

    if not query:
        return "Error: 'query' argument is required for rag_lookup."

    result = lookup_with_fallback(query, top_k=top_k)
    return result["text"]

