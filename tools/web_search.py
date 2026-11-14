import time
from ddgs import DDGS


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web and return structured search results with fallback backends.

    Args:
        query: The search query.
        max_results: Maximum number of results to return.

    Returns:
        A list of dictionaries containing title, url, snippet.
    """
    query = (query or "").strip()
    if not query:
        return []

    # Try standard 'auto' first, then fallback to 'html' or 'lite' on failure
    for attempt, backend in enumerate(("auto", "html", "lite")):
        try:
            search_results = DDGS().text(
                query,
                max_results=max_results,
                backend=backend,
            )

            results = []
            for result in search_results:
                url = result.get("href", "")
                if url:
                    results.append({
                        "title": result.get("title", ""),
                        "url": url,
                        "snippet": result.get("body", "")
                    })

            if results:
                return results

        except Exception:
            time.sleep(0.5)
            continue

    return [
        {
            "title": "",
            "url": "",
            "snippet": f"Web search failed to retrieve results for: {query}"
        }
    ]


# Tool schema for the LLM
WEB_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web and return relevant search results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}