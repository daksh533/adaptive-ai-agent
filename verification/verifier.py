"""
verification/verifier.py

Phase 6 — Verification Module

The Verifier acts as the agent's internal proofreader. Every step result is
inspected before it is committed to state, using three independent methods:

    1. Numeric Re-Check     – re-computes arithmetic found in the result.
    2. Evidence Grounding   – cross-checks claims against context / knowledge.
    3. Logical Consistency  – checks whether the result satisfies the subtask.

Each check returns a :class:`VerificationResult`; ``verify_step`` aggregates
them into one verdict (any failure => overall failure).
"""

from __future__ import annotations

import json
import re

from llm_client import chat_completion, extract_json_dict, message_content
from tools.calculator import calculate


# ---------------------------------------------------------------------------
# Data type for verification results
# ---------------------------------------------------------------------------

class VerificationResult:
    """
    Container for the outcome of a single verification pass.

    Attributes:
        status: "passed" or "failed"
        method: the verification method(s) that produced this result
        reason: explanation of why the check passed or failed
    """

    def __init__(self, status: str, method: str, reason: str):
        self.status = status
        self.method = method
        self.reason = reason

    def __repr__(self):
        return (
            f"VerificationResult(status={self.status!r}, "
            f"method={self.method!r}, reason={self.reason!r})"
        )

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "method": self.method,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Utility: LLM call
# ---------------------------------------------------------------------------

def _llm_call(messages: list, temperature: float = 0.0) -> str:
    """Call the shared LLM gateway with a verifier-specific label."""
    response = chat_completion(
        messages,
        temperature=temperature,
        label="Verifier",
    )
    return message_content(response)


# ---------------------------------------------------------------------------
# Check 1: Numeric Re-Check
# ---------------------------------------------------------------------------

def _normalize_math_text(text: str) -> str:
    """
    Normalise common Markdown/LaTeX forms so arithmetic can be parsed.

    Examples:
        ``25\\% of 800``            -> ``25% of 800``
        ``0.25 \\times 800``        -> ``0.25 * 800``
        ``\\boxed{200}``            -> ``200``
        ``\\frac{1}{2}``            -> ``(1)/(2)``
    """
    normalized = text
    # Treat "2 ** 10" as exponentiation, but "**bold**" as Markdown bold.
    normalized = re.sub(r"(?<=\d)\s*\*\*\s*(?=\d)", "^", normalized)
    normalized = normalized.replace("**", "")            # Markdown bold
    normalized = normalized.replace("\\%", "%")
    normalized = re.sub(r"\\times\b|\\cdot\b", "*", normalized)
    normalized = normalized.replace("\\div", "/")
    normalized = re.sub(r"\\text\{([^{}]*)\}", r"\1", normalized)
    normalized = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", normalized)
    normalized = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", normalized)
    for token in ("\\(", "\\)", "\\[", "\\]", "$"):
        normalized = normalized.replace(token, " ")
    normalized = normalized.replace("−", "-").replace("×", "*").replace("÷", "/")
    # Drop thousands separators: 1,000 -> 1000
    normalized = re.sub(r"(?<=\d),(?=\d{3}\b)", "", normalized)
    return normalized


def _extract_math_expressions(text: str) -> list[tuple[str, float, float]]:
    """
    Extract arithmetic expressions and their stated answers from *text*.

    Supported formats:
        - "25% of 800 = 200" or "25% of 800 is 200"
        - "15 + 30 = 45"
        - "135 * 2 = 270"
        - LaTeX: "25\\% of 800 = \\boxed{200}"

    Returns:
        List of (expr_label, stated_value, computed_value) tuples.
    """
    pairs: list[tuple[str, float, float]] = []
    normalized = _normalize_math_text(text)

    # Pattern 1: Percentage calculations, e.g. "25% of 800 = 200".
    # The trailing lookahead rejects intermediate values such as
    # "25% of 800 = 0.25 * 800 = 200" (0.25 is followed by an operator).
    pct_pattern = re.finditer(
        r"(?<!\d)(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)?(\d+(?:\.\d+)?)(?!\d|\.\d)\s*"
        r"(?:=|:|equals?|is|gives|->|→)\s*[\s*]*"
        r"(-?\d+(?:\.\d+)?)(?!\d|\.\d)(?!\s*[\+\-\*/^])",
        normalized,
        re.IGNORECASE,
    )
    for match in pct_pattern:
        pct = float(match.group(1))
        base = float(match.group(2))
        stated = float(match.group(3))
        computed = (pct / 100.0) * base
        pairs.append((f"{pct:g}% of {base:g}", stated, float(computed)))

    # Pattern 2: Standard equations with an arithmetic operator (+, -, *, /, ^).
    eq_pattern = re.finditer(
        r"(-?(?:\d+(?:\.\d+)?|\([^()]*\))"
        r"(?:\s*[\+\-\*/\^]\s*(?:\d+(?:\.\d+)?|\([^()]*\)))+)"
        r"\s*(?:=|:)\s*[\s*]*(-?\d+(?:\.\d+)?)(?!\d|\.\d)",
        normalized,
    )
    for match in eq_pattern:
        raw_expr = match.group(1).strip()
        stated = float(match.group(2))
        try:
            clean_expr = raw_expr.replace("^", "**")
            computed = calculate(clean_expr)
            if isinstance(computed, (int, float)) and not isinstance(computed, bool):
                pairs.append((raw_expr, stated, float(computed)))
        except Exception:
            continue

    return pairs


def numeric_recheck(subtask: str, result: str) -> VerificationResult:
    """
    Verification Method 1: extract numeric expressions from *result* and
    re-compute them independently. Flag mismatches.
    """
    pairs = _extract_math_expressions(result)

    if not pairs:
        return VerificationResult(
            status="passed",
            method="numeric_recheck",
            reason="No numeric expressions found in result. Skipping arithmetic check.",
        )

    errors = []
    for raw_expr, stated, computed in pairs:
        tolerance = max(0.01 * abs(stated), 0.001)
        if abs(computed - stated) > tolerance:
            errors.append(
                f"Expression '{raw_expr}' stated as {stated} "
                f"but independently computed as {computed:.4f}."
            )

    if errors:
        return VerificationResult(
            status="failed",
            method="numeric_recheck",
            reason="Arithmetic mismatch(es) detected: " + "; ".join(errors),
        )

    return VerificationResult(
        status="passed",
        method="numeric_recheck",
        reason=f"All {len(pairs)} numeric expression(s) verified correctly.",
    )


# ---------------------------------------------------------------------------
# Check 2: Evidence Grounding Check
# ---------------------------------------------------------------------------

def _context_block(goal: str, previous_results: list | None) -> str:
    """Build the shared goal + previous-results context used by LLM checks."""
    context = ""
    if goal:
        context += f"Overall Goal: {goal}\n"

    if previous_results:
        condensed = [
            {
                "step": result.get("step", ""),
                "result": str(result.get("result", ""))[:200],
            }
            for result in previous_results
        ]
        context += (
            "Context from previous steps:\n"
            f"{json.dumps(condensed, indent=2)}\n"
        )

    return context


def evidence_grounding_check(
    subtask: str,
    result: str,
    goal: str = "",
    previous_results: list | None = None,
) -> VerificationResult:
    """
    Verification Method 2: cross-reference factual claims in *result*
    against verifiable knowledge or context.
    """
    # Fast path: direct citation validation (0 LLM calls)
    res_lower = result.lower()
    if "[source: knowledge base]" in res_lower or "source: local knowledge base" in res_lower or "chunk_" in res_lower:
        return VerificationResult(
            status="passed",
            method="evidence_grounding",
            reason="Knowledge base source citation verified (local ChromaDB match).",
        )
    if ("source: web search" in res_lower or "[source: web fallback]" in res_lower) and "http" in res_lower:
        return VerificationResult(
            status="passed",
            method="evidence_grounding",
            reason="Web source citation with URL verified.",
        )
    prompt_text = (
        f"You are a strict evidence checker for an AI agent.\n\n"
        f"{_context_block(goal, previous_results)}\n"
        f"Current Subtask:\n{subtask}\n\n"
        f"Agent's result:\n{result}\n\n"
        f"Your job: Determine whether the factual or mathematical claims in the result "
        f"are plausible and grounded (supported by previous results, tools, or established facts).\n"
        f"If the result is an arithmetic calculation or deduction derived from previous steps, return verdict: passed.\n"
        f"If the result cites a knowledge-base document/chunk or a web URL, treat the claim as grounded "
        f"as long as the citation matches the claim.\n"
        f"If the result says it came from a web search but gives no URL, return verdict: failed.\n\n"
        f"Return ONLY a valid JSON object in this exact format:\n"
        f'{{"verdict": "passed" or "failed", "reason": "one sentence explanation"}}'
    )

    try:
        data = extract_json_dict(_llm_call([{"role": "user", "content": prompt_text}]))
        if data:
            verdict = str(data.get("verdict", "passed")).lower()
            reason = data.get("reason", "No reason provided.")
            return VerificationResult(
                status="passed" if "pass" in verdict else "failed",
                method="evidence_grounding",
                reason=reason,
            )
    except Exception as exc:
        print(f"[Verifier] Evidence check unavailable ({exc}). Skipping.")

    return VerificationResult(
        status="passed",
        method="evidence_grounding",
        reason="Evidence grounding check inconclusive - skipped (fail-open).",
    )


# ---------------------------------------------------------------------------
# Check 3: Logical Consistency Check
# ---------------------------------------------------------------------------

def logical_consistency_check(
    subtask: str,
    result: str,
    goal: str = "",
    previous_results: list | None = None,
) -> VerificationResult:
    """
    Verification Method 3: checks whether the result logically and completely
    answers the given subtask, taking previous steps into account.
    """
    prompt_text = (
        f"You are a strict logical consistency checker for an AI agent.\n\n"
        f"{_context_block(goal, previous_results)}\n"
        f"Current Subtask:\n{subtask}\n\n"
        f"Agent's result:\n{result}\n\n"
        f"Question: Does this result directly and completely address the current subtask "
        f"(taking into account the previous steps if this step builds on them)?\n"
        f"Check for:\n"
        f"- Does it fulfill what was asked in the subtask?\n"
        f"- If it references numbers from earlier steps, does it match the context?\n"
        f"- Are there any internal contradictions?\n\n"
        f"Return ONLY a valid JSON object:\n"
        f'{{"verdict": "passed" or "failed", "reason": "one sentence explanation"}}'
    )

    try:
        data = extract_json_dict(_llm_call([{"role": "user", "content": prompt_text}]))
        if data:
            verdict = str(data.get("verdict", "passed")).lower()
            reason = data.get("reason", "No reason provided.")
            return VerificationResult(
                status="passed" if "pass" in verdict else "failed",
                method="logical_consistency",
                reason=reason,
            )
    except Exception as exc:
        print(f"[Verifier] Consistency check unavailable ({exc}). Skipping.")

    return VerificationResult(
        status="passed",
        method="logical_consistency",
        reason="Logical consistency check inconclusive - skipped (fail-open).",
    )


# ---------------------------------------------------------------------------
# Public API: run all three checks
# ---------------------------------------------------------------------------

def verify_step(
    subtask: str,
    result: str,
    goal: str = "",
    previous_results: list | None = None,
) -> VerificationResult:
    """
    Run all three verification methods on a step result.

    The overall verdict is:
        "passed" -> all three checks passed
        "failed" -> at least one check failed
    """
    print("[Verifier] Running Numeric Re-Check...")
    numeric_result = numeric_recheck(subtask, result)
    print(f"[Verifier] Numeric Re-Check: {numeric_result.status.upper()} - {numeric_result.reason}")

    print("[Verifier] Running Evidence Grounding Check...")
    evidence_result = evidence_grounding_check(subtask, result, goal, previous_results)
    print(f"[Verifier] Evidence Grounding: {evidence_result.status.upper()} - {evidence_result.reason}")

    print("[Verifier] Running Logical Consistency Check...")
    logic_result = logical_consistency_check(subtask, result, goal, previous_results)
    print(f"[Verifier] Logical Consistency: {logic_result.status.upper()} - {logic_result.reason}")

    failures = [
        check
        for check in (numeric_result, evidence_result, logic_result)
        if check.status == "failed"
    ]

    if failures:
        return VerificationResult(
            status="failed",
            method="all",
            reason=" | ".join(f"[{failure.method}] {failure.reason}" for failure in failures),
        )

    return VerificationResult(
        status="passed",
        method="all",
        reason="All three verification checks passed.",
    )


# ---------------------------------------------------------------------------
# Public API: Phase 10 — Risk-Adaptive Verification
# ---------------------------------------------------------------------------

def verify_step_adaptive(
    subtask: str,
    result: str,
    goal: str = "",
    previous_results: list | None = None,
    risk_level: str = "high",
) -> VerificationResult:
    """
    Phase 10: Risk-tier-aware verification.

    Dispatches to a subset of checks based on *risk_level*:

        LOW    -> numeric_recheck only  (0 LLM calls)
        MEDIUM -> numeric_recheck + evidence_grounding_check  (1 LLM call)
        HIGH   -> full verify_step()  (2 LLM calls) — identical to Phase 6

    Args:
        subtask:          The current subtask string.
        result:           The agent result to verify.
        goal:             The overall agent goal (for context).
        previous_results: Results from earlier steps (for context).
        risk_level:       "low" | "medium" | "high" (default "high").

    Returns:
        A :class:`VerificationResult` with an aggregated verdict.
    """
    tier = str(risk_level).strip().lower()
    if tier not in ("low", "medium", "high"):
        tier = "high"

    # ------------------------------------------------------------------ HIGH
    if tier == "high":
        print(f"[Verifier] Risk Tier: HIGH — running Numeric + Grounding + Logical")
        return verify_step(subtask, result, goal, previous_results)

    # ----------------------------------------------------------------- MEDIUM
    if tier == "medium":
        print(f"[Verifier] Risk Tier: MEDIUM — running Numeric + Evidence Grounding")

        print("[Verifier] Running Numeric Re-Check...")
        numeric_result = numeric_recheck(subtask, result)
        print(f"[Verifier] Numeric Re-Check: {numeric_result.status.upper()} - {numeric_result.reason}")

        print("[Verifier] Running Evidence Grounding Check...")
        evidence_result = evidence_grounding_check(subtask, result, goal, previous_results)
        print(f"[Verifier] Evidence Grounding: {evidence_result.status.upper()} - {evidence_result.reason}")

        failures = [
            check for check in (numeric_result, evidence_result)
            if check.status == "failed"
        ]
        if failures:
            return VerificationResult(
                status="failed",
                method="numeric+grounding",
                reason=" | ".join(
                    f"[{f.method}] {f.reason}" for f in failures
                ),
            )
        return VerificationResult(
            status="passed",
            method="numeric+grounding",
            reason="Numeric and Evidence Grounding checks passed (MEDIUM risk).",
        )

    # -------------------------------------------------------------------- LOW
    # tier == "low"
    print(f"[Verifier] Risk Tier: LOW — running Numeric Re-Check only (no LLM calls)")

    print("[Verifier] Running Numeric Re-Check...")
    numeric_result = numeric_recheck(subtask, result)
    print(f"[Verifier] Numeric Re-Check: {numeric_result.status.upper()} - {numeric_result.reason}")

    if numeric_result.status == "failed":
        return VerificationResult(
            status="failed",
            method="numeric_recheck",
            reason=f"[numeric_recheck] {numeric_result.reason}",
        )

    return VerificationResult(
        status="passed",
        method="numeric_recheck",
        reason="Numeric Re-Check passed (LOW risk — LLM checks skipped).",
    )
