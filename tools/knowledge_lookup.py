"""
tools/knowledge_lookup.py

RAG-first lookup with automatic web fallback.

Workflow (all inside one tool call):

    1. Search the local ChromaDB knowledge base.
    2. If at least one chunk is sufficiently relevant, answer from the KB.
    3. Otherwise (no chunks, low relevance, or the KB is unavailable),
       search the live web with DuckDuckGo.
    4. Label the result clearly:

           [SOURCE: KNOWLEDGE BASE]
           [SOURCE: WEB FALLBACK]
           [SOURCE: UNAVAILABLE]

The label lets the executor tell the user *where* the information came from,
which is exactly the workflow we want:

    "If the knowledge base has it, use the knowledge base.
     If it does not, search the web and say so."

Environment:
    RAG_MAX_DISTANCE  – maximum Chroma distance still considered relevant
                        (default: 0.75; lower = stricter)
"""

from __future__ import annotations

import os

from rag.retriever import retrieve
from tools.web_search import web_search


DEFAULT_TOP_K = 4
DEFAULT_MAX_WEB_RESULTS = 3

SOURCE_KB = "knowledge_base"
SOURCE_WEB = "web"
SOURCE_NONE = "none"


def _max_distance() -> float:
    """Read the relevance threshold at call time so .env changes apply."""
    try:
        return float(os.getenv("RAG_MAX_DISTANCE", "0.75"))
    except (TypeError, ValueError):
        return 0.75


def _is_relevant(chunk: dict, max_distance: float) -> bool:
    try:
        return float(chunk.get("distance", 999.0)) <= max_distance
    except (TypeError, ValueError):
        return False


def _format_kb_results(query: str, chunks: list[dict]) -> str:
    lines = [
        "[SOURCE: KNOWLEDGE BASE]",
        f"The local knowledge base contains relevant information for: '{query}'.",
        "Answer from the chunks below and cite the document name and chunk number.",
        "",
    ]
    for index, chunk in enumerate(chunks, start=1):
        lines.append(
            f"--- Knowledge base result {index} ---\n"
            f"Source: {chunk.get('source', 'unknown')} "
            f"(chunk {chunk.get('chunk_index', -1)} of "
            f"{chunk.get('total_chunks', -1)})\n"
            f"Distance: {chunk.get('distance', 'n/a')}\n"
            f"Content:\n{chunk.get('text', '')}\n"
        )
    return "\n".join(lines)


def _format_web_results(query: str, reason: str, results: list[dict]) -> str:
    lines = [
        "[SOURCE: WEB FALLBACK]",
        "IMPORTANT: Tell the user that this information was NOT found in the "
        "local knowledge base and therefore comes from a live web search.",
        f"Reason for fallback: {reason}",
        f"Query: '{query}'",
        "Answer from these results and include at least one URL as the source.",
        "",
    ]
    for index, result in enumerate(results, start=1):
        lines.append(
            f"--- Web result {index} ---\n"
            f"Title: {result.get('title', '')}\n"
            f"URL: {result.get('url', '')}\n"
            f"Snippet: {result.get('snippet', '')}\n"
        )
    return "\n".join(lines)


def lookup_with_fallback(
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    max_web_results: int = DEFAULT_MAX_WEB_RESULTS,
) -> dict:
    """
    Search the knowledge base first, then fall back to the web when needed.

    Args:
        query:           Natural-language search query.
        top_k:           Number of knowledge-base chunks to request.
        max_web_results: Number of web results to return on fallback.

    Returns:
        A dict with:
            source:       "knowledge_base" | "web" | "none"
            text:         LLM-ready, source-labelled context
            chunks:       knowledge-base chunks that were used (may be empty)
            web_results:  web results that were used (may be empty)
            reason:       why the fallback (or failure) happened
    """
    query = (query or "").strip()
    if not query:
        return {
            "source": SOURCE_NONE,
            "text": "Error: 'query' is required for knowledge_lookup.",
            "chunks": [],
            "web_results": [],
            "reason": "empty query",
        }

    max_distance = _max_distance()

    # --- Step 1: local knowledge base -------------------------------------
    chunks: list[dict] = []
    rag_error = None
    try:
        chunks = retrieve(query, top_k=top_k)
    except Exception as exc:  # missing store, embedding failure, ...
        rag_error = str(exc)

    relevant = [chunk for chunk in chunks if _is_relevant(chunk, max_distance)]

    if relevant:
        return {
            "source": SOURCE_KB,
            "text": _format_kb_results(query, relevant),
            "chunks": relevant,
            "web_results": [],
            "reason": "",
        }

    # --- Step 2: explain why we are falling back --------------------------
    if rag_error:
        reason = f"local knowledge base unavailable ({rag_error})"
    elif not chunks:
        reason = "local knowledge base returned no matching chunks"
    else:
        best = min(
            (float(chunk.get("distance", 999.0)) for chunk in chunks),
            default=999.0,
        )
        reason = (
            f"best knowledge-base match had distance {best:.4f}, "
            f"above the relevance threshold {max_distance:.4f}"
        )

    # --- Step 3: live web fallback ---------------------------------------
    try:
        web_results = web_search(query, max_results=max_web_results)
    except Exception as exc:
        web_results = [
            {
                "title": "",
                "url": "",
                "snippet": f"Web search failed: {exc}",
            }
        ]

    usable = [
        result
        for result in web_results
        if result.get("url")
        and not str(result.get("snippet", "")).startswith("Web search failed")
    ]

    if usable:
        return {
            "source": SOURCE_WEB,
            "text": _format_web_results(query, reason, usable),
            "chunks": [],
            "web_results": usable,
            "reason": reason,
        }

    detail = (
        usable[0].get("snippet")
        if usable
        else (web_results[0].get("snippet") if web_results else "no web results")
    )
    return {
        "source": SOURCE_NONE,
        "text": (
            "[SOURCE: UNAVAILABLE]\n"
            f"The local knowledge base had no relevant information ({reason}), "
            f"and the web search also failed ({detail}).\n"
            "Tell the user that the answer could not be verified from either source."
        ),
        "chunks": [],
        "web_results": [],
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Tool schema (OpenAI function-calling format)
# ---------------------------------------------------------------------------

KNOWLEDGE_LOOKUP_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "knowledge_lookup",
        "description": (
            "PREFERRED tool for factual or document questions. It searches "
            "the local knowledge base first and automatically falls back to "
            "a live web search only when the knowledge base has no relevant "
            "information. The output is labelled [SOURCE: KNOWLEDGE BASE], "
            "[SOURCE: WEB FALLBACK], or [SOURCE: UNAVAILABLE] so you can tell "
            "the user exactly where the answer came from. Always cite the "
            "document/chunk or the web URL in your final answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The natural-language question or search query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Knowledge-base chunks to retrieve (default 4).",
                    "default": 4,
                },
                "max_web_results": {
                    "type": "integer",
                    "description": "Web results to return if fallback is needed (default 3).",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
}


def handle_tool_call(arguments: dict) -> str:
    """Execute a knowledge_lookup tool call and return labelled context."""
    if not isinstance(arguments, dict):
        return "Error: Tool arguments must be a dictionary."

    query = str(arguments.get("query", "")).strip()
    if not query:
        return "Error: 'query' argument is required for knowledge_lookup."

    try:
        top_k = int(arguments.get("top_k", DEFAULT_TOP_K))
    except (TypeError, ValueError):
        top_k = DEFAULT_TOP_K
    top_k = max(1, min(top_k, 10))

    try:
        max_web_results = int(
            arguments.get("max_web_results", DEFAULT_MAX_WEB_RESULTS)
        )
    except (TypeError, ValueError):
        max_web_results = DEFAULT_MAX_WEB_RESULTS
    max_web_results = max(1, min(max_web_results, 5))

    result = lookup_with_fallback(
        query,
        top_k=top_k,
        max_web_results=max_web_results,
    )
    return result["text"]
