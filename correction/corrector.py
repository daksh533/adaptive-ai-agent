"""
correction/corrector.py

Phase 8 — Targeted Self-Correction Engine

Maintains one recovery strategy per diagnosed error class:
    calculation_error -> re-compute with the calculator tool
    tool_error        -> retry with reformulated tool calls
    grounding_error   -> search for evidence before answering
    consistency_error -> rewrite for internal consistency
    incomplete_error  -> expand the answer to cover the subtask
    unknown_error     -> simplified retry
"""

from __future__ import annotations

import json
from typing import Any

from llm_client import chat_completion, message_content
from tools.calculator import CALCULATOR_TOOL_SCHEMA
from tools.registry import TOOL_SCHEMAS, dispatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm(messages: list, temperature: float = 0.2) -> str:
    response = chat_completion(
        messages,
        temperature=temperature,
        label="Corrector",
    )
    return message_content(response)


def _parse_arguments(raw: str | None) -> dict:
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _previous_context(previous_results: list[dict], limit: int = 5) -> str:
    if not previous_results:
        return ""
    condensed = []
    for r in previous_results[-limit:]:
        condensed.append(
            {
                "step": r.get("step", ""),
                "result": str(r.get("result", ""))[:400],
            }
        )
    return (
        "\nAlready completed and verified previous steps (DO NOT re-do these):\n"
        + json.dumps(condensed, indent=2)[:1500]
        + "\n"
    )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _correct_calculation(subtask: str, result: str, previous_results: list[dict]) -> str:
    context = _previous_context(previous_results)
    prompt = (
        f"The previous calculation result was incorrect or unverifiable.\n"
        f"{context}"
        f"Current Subtask: {subtask}\nPrevious result: {result}\n"
        f"Do NOT re-do previous steps. Use the completed results above if needed.\n"
        f"Recompute using the calculate tool and return ONLY the final numeric answer.\n"
    )
    response = chat_completion(
        [
            {"role": "system", "content": "You are a math correction assistant. Use the calculate tool."},
            {"role": "user", "content": prompt},
        ],
        tools=[CALCULATOR_TOOL_SCHEMA],
        tool_choice="auto",
        label="Corrector",
    )

    message = response.choices[0].message
    if message.tool_calls:
        arguments = _parse_arguments(message.tool_calls[0].function.arguments)
        return str(dispatch("calculate", arguments))
    return message_content(response) or "Calculation correction failed."


def _tool_retry_loop(messages: list, max_turns: int = 2) -> list:
    """Run up to *max_turns* tool-calling turns and append results to messages."""
    for _ in range(max_turns):
        response = chat_completion(
            messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            label="Corrector",
        )
        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            break

        for tool_call in message.tool_calls:
            arguments = _parse_arguments(tool_call.function.arguments)
            try:
                tool_output = dispatch(tool_call.function.name, arguments)
            except Exception as exc:
                tool_output = f"Error: {exc}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_output)[:1000],
                }
            )

    return messages


def _correct_tool(subtask: str, result: str, previous_results: list[dict]) -> str:
    context = _previous_context(previous_results)
    prompt = (
        f"A tool call failed or gave an incorrect result while executing this subtask.\n"
        f"{context}"
        f"Current Subtask: {subtask}\nError/result: {result}\n"
        f"CRITICAL: Do NOT re-do any previous steps. Use the verified results above.\n"
        f"Reformulate and provide the corrected final answer using available tools "
        f"(knowledge_lookup, rag_lookup, web_search, run_python, calculate). "
        f"Once information is obtained, synthesize the final answer text immediately without looping.\n"
    )
    messages = [
        {"role": "system", "content": "You are a tool-correction assistant. Fix the tool call for the current subtask without re-doing earlier steps."},
        {"role": "user", "content": prompt},
    ]
    messages = _tool_retry_loop(messages)
    last_msg = messages[-1]
    if hasattr(last_msg, "content") and last_msg.content and last_msg.content.strip():
        return last_msg.content.strip()

    final = chat_completion(
        messages + [{"role": "user", "content": "Provide your final answer now in plain text without making any more tool calls."}],
        tools=None,
        label="Corrector",
    )
    return message_content(final) or "Tool correction failed."


def _correct_grounding(subtask: str, result: str, previous_results: list[dict]) -> str:
    context = _previous_context(previous_results)
    prompt = (
        f"The result lacks evidence grounding.\n"
        f"{context}"
        f"Current Subtask: {subtask}\nPrevious result: {result}\n"
        f"CRITICAL: Do NOT re-do any previous steps. Search for supporting evidence for this specific subtask "
        f"using knowledge_lookup, rag_lookup, or web_search, then provide a grounded final answer with citations or URLs.\n"
    )
    messages = [
        {"role": "system", "content": "You are a grounding assistant. Find evidence for this subtask without repeating previous steps."},
        {"role": "user", "content": prompt},
    ]
    messages = _tool_retry_loop(messages, max_turns=1)
    last_msg = messages[-1]
    if hasattr(last_msg, "content") and last_msg.content and last_msg.content.strip():
        return last_msg.content.strip()

    final = chat_completion(
        messages + [{"role": "user", "content": "Give the final grounded answer with sources in plain text without making any more tool calls."}],
        tools=None,
        label="Corrector",
    )
    return message_content(final) or "Grounding correction failed."


def _correct_consistency(subtask: str, result: str, previous_results: list[dict]) -> str:
    context = _previous_context(previous_results)
    prompt = (
        f"The result has logical consistency issues.\n"
        f"Subtask: {subtask}\nPrevious result: {result}\n{context}\n"
        f"Rewrite the result to be internally consistent and fully address the subtask.\n"
    )
    return _llm(
        [
            {"role": "system", "content": "You are a consistency editor. Fix logical issues in the answer."},
            {"role": "user", "content": prompt},
        ]
    )


def _correct_incomplete(subtask: str, result: str, previous_results: list[dict]) -> str:
    context = _previous_context(previous_results)
    prompt = (
        f"The result is incomplete for this subtask.\n"
        f"Subtask: {subtask}\nPrevious result: {result}\n{context}\n"
        f"Expand the answer to fully address every part of the subtask.\n"
    )
    return _llm(
        [
            {"role": "system", "content": "You are a completion assistant. Make the answer complete."},
            {"role": "user", "content": prompt},
        ]
    )


def _correct_unknown(subtask: str, result: str, previous_results: list[dict]) -> str:
    context = _previous_context(previous_results)
    prompt = (
        f"Retry this subtask with a simplified, direct approach.\n"
        f"Subtask: {subtask}\nPrevious result: {result}\n{context}\n"
        f"Provide the best possible answer.\n"
    )
    return _llm(
        [
            {"role": "system", "content": "You are a general correction assistant. Retry the subtask simply."},
            {"role": "user", "content": prompt},
        ]
    )


_STRATEGIES = {
    "calculation_error": _correct_calculation,
    "tool_error": _correct_tool,
    "grounding_error": _correct_grounding,
    "consistency_error": _correct_consistency,
    "incomplete_error": _correct_incomplete,
    "unknown_error": _correct_unknown,
}


def correct_step(
    subtask: str,
    result: str,
    error_class: str,
    previous_results: list[dict],
) -> dict[str, Any]:
    """
    Run the correction strategy for the diagnosed error class.

    Returns:
        ``{"corrected_result": str, "correction_strategy": str}``.
    """
    strategy = _STRATEGIES.get(error_class, _correct_unknown)
    try:
        corrected = strategy(subtask, result, previous_results)
    except Exception as exc:
        corrected = f"Correction failed: {exc}"

    return {
        "corrected_result": corrected,
        "correction_strategy": error_class,
    }
