"""
agent/planner.py

Phase 3/4 — Task planning.

Converts the user's goal into an ordered list of subtasks using the shared
LLM gateway. The planner is intentionally conservative: it must return a JSON
object with a non-empty ``plan`` list of strings.
"""

from llm_client import chat_completion, extract_plan, message_content


SYSTEM_PROMPT = (
    "You are a task planner for an AI agent. "
    "Break the user's goal into a small number of clear, "
    "ordered subtasks.\n\n"

    "Return ONLY valid JSON in exactly this format:\n"
    "{"
    "\"plan\": ["
    "\"step 1\", "
    "\"step 2\", "
    "\"step 3\""
    "]"
    "}\n\n"

    "Rules:\n"
    "1. Each item must be one clear subtask.\n"
    "2. Put subtasks in the correct execution order.\n"
    "3. Later steps may depend on results from earlier steps.\n"
    "4. The agent uses built-in programmatic tools: calculate, web_search, "
    "run_python, and knowledge_lookup/rag_lookup (which searches the local knowledge base first and automatically falls back to live web search). "
    "Do not create separate redundant steps for KB and web; create a single lookup step.\n"
    "5. Do not create subtasks asking to open external desktop apps or GUI programs.\n"
    "6. Do not perform the task yourself.\n"
    "7. Do not include explanations outside the JSON."
)


def create_plan(goal: str) -> list[str]:
    """
    Convert the user's goal into an ordered list of subtasks.

    Raises:
        ValueError: if the planner returns an invalid or empty plan.
    """
    response = chat_completion(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": goal},
        ],
        temperature=0.2,
        label="Planner",
    )

    plan = extract_plan(message_content(response))
    if plan is None:
        raise ValueError("Planner returned an invalid plan.")
    return plan
