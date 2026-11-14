"""
planner/replanner.py

Phase 9 — Dynamic Plan Invalidation & Replanning

Triggered when a step fails after the maximum number of targeted corrections.
Recomputes the remaining subtasks conditioned on what has already been
completed, and keeps a fallback to the original remaining plan if the LLM
returns malformed JSON.
"""

from __future__ import annotations

from llm_client import chat_completion, extract_plan, message_content


def replan_remaining(
    goal: str,
    completed_results: list,
    remaining_subtasks: list,
    failure_reason: str,
) -> list:
    """
    Recompute remaining subtasks given what has already been completed.

    Args:
        goal:              Original user goal.
        completed_results: Passed results only.
        remaining_subtasks: Subtasks not yet executed (includes the failed one).
        failure_reason:    Why the original plan is being invalidated.

    Returns:
        A new ordered list of subtask strings (strictly remaining steps only).
    """
    completed_text = ""
    if completed_results:
        completed_text = "ALREADY COMPLETED STEPS (DO NOT REPEAT THESE):\n"
        for i, result in enumerate(completed_results, 1):
            completed_text += (
                f"- Step {i}: {result.get('step', '')}\n"
                f"  Result: {str(result.get('result', ''))[:300]}\n"
            )

    remaining_text = "Subtasks that were pending or failed:\n"
    for index, step in enumerate(remaining_subtasks, 1):
        remaining_text += f"{index}. {step}\n"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a dynamic replanner for an AI agent.\n"
                "A plan failed partway through execution.\n"
                "CRITICAL RULES:\n"
                "1. DO NOT repeat or re-execute ANY of the already completed steps.\n"
                "2. The completed steps are ALREADY FINISHED, verified, and saved.\n"
                "3. Provide ONLY the new/revised subtasks needed to finish the UNCOMPLETED work.\n"
                "4. Do NOT start from step 1 of the original goal.\n"
                "5. Return ONLY valid JSON: {\"plan\": [\"remaining subtask 1\", \"remaining subtask 2\"]}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Goal: {goal}\n\n"
                f"{completed_text}\n"
                f"{remaining_text}\n"
                f"Failure reason: {failure_reason}\n\n"
                "Provide ONLY the remaining subtasks to complete the goal:"
            ),
        },
    ]

    try:
        response = chat_completion(messages, temperature=0.3, label="Replanner")
        plan = extract_plan(message_content(response))
        if plan:
            # Deduplicate against already completed steps
            completed_steps_lower = [
                str(r.get("step", "")).lower().strip() for r in completed_results
            ]
            clean_plan = []
            for step in plan:
                step_lower = step.lower().strip()
                # Check if this step is practically identical to an already completed step
                is_duplicate = any(
                    comp == step_lower
                    or (len(comp) > 15 and comp in step_lower)
                    for comp in completed_steps_lower
                )
                if not is_duplicate:
                    clean_plan.append(step)

            if clean_plan:
                return clean_plan
    except Exception as exc:
        print(f"[Replanner] LLM replan unavailable ({exc}). Keeping remaining plan.")

    # Fallback: return the remaining subtasks unchanged.
    return list(remaining_subtasks)


def replan_step(
    goal: str,
    completed_results: list,
    remaining_subtasks: list,
    failure_reason: str,
) -> dict:
    """Public API: returns a dict suitable for merging into GraphState."""
    new_plan = replan_remaining(goal, completed_results, remaining_subtasks, failure_reason)
    return {
        "plan": new_plan,
        "current_step": 0,
        "is_done": not new_plan,
    }
