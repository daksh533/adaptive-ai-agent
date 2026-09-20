from .calculator import (
    CALCULATOR_TOOL_SCHEMA,
    handle_tool_call as handle_calculator
)

from .web_search import (
    WEB_SEARCH_TOOL_SCHEMA,
    web_search
)

from .python_tools import (
    PYTHON_TOOL_SCHEMA,
    handle_tool_call as handle_python
)


# All tools available to the LLM
TOOL_SCHEMAS = [
    CALCULATOR_TOOL_SCHEMA,
    WEB_SEARCH_TOOL_SCHEMA,
    PYTHON_TOOL_SCHEMA,
]


def dispatch(name: str, arguments: dict):
    """
    Send a tool call to the correct tool.

    Args:
        name: Name of the tool requested by the LLM.
        arguments: Arguments provided by the LLM.

    Returns:
        Result returned by the selected tool.
    """

    if name == "calculate":
        return handle_calculator(arguments)

    elif name == "web_search":
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 5)

        return web_search(
            query=query,
            max_results=max_results
        )

    elif name == "run_python":
        return handle_python(arguments)

    else:
        return f"Error: Unknown tool '{name}'"