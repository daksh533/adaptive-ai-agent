"""
agent/executor.py

Phase 2/4 — Subtask execution.

Given one subtask and the results of previous steps, ask the LLM to complete
it. The executor supports up to ``MAX_TOOL_TURNS`` rounds of tool calls and
then forces a plain-text final answer.
"""

import json

from openai import BadRequestError

from llm_client import chat_completion
from tools.registry import TOOL_SCHEMAS, dispatch


MAX_TOOL_TURNS = 3
MAX_RETRIES = 3


SYSTEM_PROMPT = (
    "You are the executor of an AI agent.\n\n"

    "Your job is to execute ONE subtask at a time.\n\n"

    "You have access to these tools:\n"
    "- knowledge_lookup: PREFERRED for factual, document, project, or "
    "'who/what/where' questions. It searches the local knowledge base FIRST "
    "and automatically falls back to live web search when the knowledge base "
    "has no relevant information. Parameter: {'query': '<search query>'}\n"
    "- rag_lookup: Knowledge search tool. Also checks the local knowledge base "
    "first and automatically falls back to web search if not found. Parameter: {'query': '<search query>'}\n"
    "- calculate: mathematical calculations. Parameter: {'expression': '<math expr>'}\n"
    "- web_search: explicit internet search. Parameter: {'query': '<search query>'}\n"
    "- run_python: Python computation or data processing. Parameter: {'code': '<python code string>'}\n"
    "  IMPORTANT for run_python: Put the Python code in the 'code' parameter string, "
    "e.g. {\"code\": \"print(...) \"}. Do NOT put answers or stdout in arguments.\n\n"

    "RAG-FIRST RESEARCH WORKFLOW (follow strictly):\n"
    "1. For factual or lookup questions, call knowledge_lookup (or rag_lookup).\n"
    "2. The tool checks the knowledge base first. If the information is not present, "
    "it automatically searches the live web and returns web results.\n"
    "3. If the tool says [SOURCE: KNOWLEDGE BASE], answer using it, cite the "
    "document and chunk number, and state: Source: local knowledge base.\n"
    "4. If the tool says [SOURCE: WEB FALLBACK], answer using it, include at "
    "least one URL, and state: 'This information was not found in the local knowledge base, "
    "so I retrieved it from web search.' End with: Source: web search (<url>).\n"
    "5. CRITICAL: Once the tool returns information (from either KB or web), "
    "TAKE THAT INFORMATION AND PROVIDE YOUR FINAL ANSWER IMMEDIATELY. "
    "Do NOT call search again for the same query. Do NOT stay stuck on this step.\n"
    "6. If the tool says [SOURCE: UNAVAILABLE], state that the answer could not be "
    "found in either the knowledge base or web search. Do not invent an answer.\n\n"

    "Choose the appropriate tool when needed.\n"
    "Use previous step results when the current subtask depends on them.\n"
    "Do not try to execute future subtasks.\n"
    "Complete only the current subtask."
)


def _format_previous_results(previous_results: list[dict]) -> str:
    """Condense earlier step results into a compact context block."""
    if not previous_results:
        return ""

    condensed = []
    for result in previous_results:
        text = str(result.get("result", ""))
        if len(text) > 500:
            text = text[:500] + "... [truncated]"
        condensed.append(
            {
                "step": result.get("step", ""),
                "result": text,
            }
        )

    return "\n\nResults from previous steps:\n" + json.dumps(condensed, indent=2)


def execute_step(subtask: str, previous_results: list[dict]) -> dict:
    """
    Execute one subtask using the available tools.

    Args:
        subtask:          The subtask description.
        previous_results: Results of earlier steps, used as context.

    Returns:
        ``{"step": subtask, "result": <text>}``.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Current subtask:\n{subtask}{_format_previous_results(previous_results)}",
        },
    ]

    retries = 0
    tool_turns = 0

    while retries < MAX_RETRIES:
        tools_allowed = tool_turns < MAX_TOOL_TURNS

        try:
            response = chat_completion(
                messages,
                tools=TOOL_SCHEMAS if tools_allowed else None,
                label="Executor",
            )
        except BadRequestError as exc:
            retries += 1
            print(
                f"[Executor] Tool call formatting error: {exc}. "
                f"Retrying step ({retries}/{MAX_RETRIES})..."
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Please provide the final result for this subtask in "
                        "plain text without malformed tool arguments."
                    ),
                }
            )
            continue

        message = response.choices[0].message

        if tools_allowed and message.tool_calls:
            tool_turns += 1
            messages.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name

                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except Exception:
                    arguments = {"code": tool_call.function.arguments}

                print(f"[Executor] Tool used: {tool_name}")

                try:
                    tool_result = dispatch(tool_name, arguments)
                except Exception as exc:  # a broken tool must not crash the run
                    tool_result = f"Error executing tool '{tool_name}': {exc}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(tool_result)[:4000],
                    }
                )

            continue

        result = (message.content or "").strip()
        if result:
            return {"step": subtask, "result": result}

        # Empty answer — nudge the model once more.
        retries += 1
        messages.append(
            {
                "role": "user",
                "content": "Please provide your complete final answer for this subtask.",
            }
        )

    return {
        "step": subtask,
        "result": "Error: Exceeded maximum tool execution retries.",
    }
