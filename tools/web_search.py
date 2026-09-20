from ddgs import DDGS


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web and return structured search results.

    Args:
        query: The search query.
        max_results: Maximum number of results to return.

    Returns:
        A list of dictionaries containing:
        - title
        - url
        - snippet
    """

    try:
        search_results = DDGS().text(
            query,
            max_results=max_results,
            backend="auto"
        )

        results = []

        for result in search_results:
            results.append({
                "title": result.get("title", ""),
                "url": result.get("href", ""),
                "snippet": result.get("body", "")
            })

        return results

    except Exception as e:
        return [
            {
                "title": "",
                "url": "",
                "snippet": f"Web search failed: {str(e)}"
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