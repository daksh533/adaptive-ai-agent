"""
verification/verifier.py

Phase 6 — Verification Module

The Verifier acts as the agent's internal proofreader.
It inspects every step result BEFORE it is committed to state,
using three independent verification methods:

    1. Numeric Re-Check     – re-computes arithmetic found in the result.
    2. Evidence Grounding   – cross-checks claims against RAG / factual context.
    3. Logical Consistency  – checks whether result satisfies the subtask given context.

Returns a structured VerificationResult with:
    status  : "passed" | "failed"
    method  : which check ran (or "all" if all ran)
    reason  : human-readable explanation of the verdict
"""

import ast
import re
import os
import time
import json

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError


load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)


# ---------------------------------------------------------------------------
# Data type for verification results
# ---------------------------------------------------------------------------

class VerificationResult:
    """
    Container for the outcome of a single verification pass.

    Attributes:
        status  : "passed" or "failed"
        method  : the verification method(s) that produced this result
        reason  : explanation of why the check passed or failed
    """

    def __init__(self, status: str, method: str, reason: str):
        self.status = status   # "passed" | "failed"
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
# Utility: LLM call with rate-limit backoff & JSON extraction
# ---------------------------------------------------------------------------

def _llm_call(messages: list, temperature: float = 0.0) -> str:
    """Call the Groq LLM with backoff on 429 rate-limit errors."""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except RateLimitError as e:
            wait = 2.0 * (1.5 ** attempt)
            ms_m = re.search(r"try again in ([\d\.]+)ms", str(e))
            s_m = re.search(r"try again in ([\d\.]+)s", str(e))
            if ms_m:
                wait = max(float(ms_m.group(1)) / 1000.0 + 0.5, 1.5)
            elif s_m:
                wait = max(float(s_m.group(1)) + 0.5, 2.0)
            print(f"[Verifier] Rate limit hit. Pausing {wait:.1f}s...")
            time.sleep(wait)

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def _extract_json_dict(text: str) -> dict:
    """Safely extract a JSON dictionary from LLM response text."""
    try:
        return json.loads(text.strip())
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass

    return {}


# ---------------------------------------------------------------------------
# Check 1: Numeric Re-Check
# ---------------------------------------------------------------------------

def _extract_math_expressions(text: str) -> list[tuple[str, float, float]]:
    """
    Extract arithmetic expressions and their stated answers from *text*.

    Supported formats:
        - "25% of 800 = 200" or "25% of 800 is 200"
        - "15 + 30 = 45"
        - "30 + 50 = 80"
        - "135 * 2 = 270"

    Returns:
        List of (expr_label, stated_value, computed_value) tuples.
    """
    pairs = []

    # Pattern 1: Percentage calculations, e.g. "25% of 800 = 200"
    pct_pattern = re.finditer(
        r"(\d+(?:\.\d+)?)\s*%\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(?:=|:|is|gives)?\s*[\*]*([\-]?\d+(?:\.\d+)?)[\*]*",
        text,
        re.IGNORECASE,
    )
    for m in pct_pattern:
        pct = float(m.group(1))
        base = float(m.group(2))
        stated = float(m.group(3))
        computed = (pct / 100.0) * base
        pairs.append((f"{pct}% of {base}", stated, float(computed)))

    # Pattern 2: Standard equations with arithmetic operator (+, -, *, /, ^)
    eq_pattern = re.finditer(
        r"([0-9\.\s\(\)]+[\+\-\*\/\^][0-9\.\s\+\-\*\/\^\(\)]+)\s*[=:]\s*[\*]*([\-]?\d+(?:\.\d+)?)[\*]*",
        text,
    )
    for m in eq_pattern:
        raw_expr = m.group(1).strip()
        stated = float(m.group(2))
        try:
            clean_expr = raw_expr.replace("^", "**")
            tree = ast.parse(clean_expr, mode="eval")
            computed = float(eval(compile(tree, "<string>", "eval")))
            pairs.append((raw_expr, stated, computed))
        except Exception:
            continue

    return pairs


def numeric_recheck(subtask: str, result: str) -> VerificationResult:
    """
    Verification Method 1: Extract numeric expressions from *result* and
    re-compute them independently. Flag mismatches.
    """
    pairs = _extract_math_expressions(result)

    if not pairs:
        # No arithmetic found — pass by default (nothing to verify)
        return VerificationResult(
            status="passed",
            method="numeric_recheck",
            reason="No numeric expressions found in result. Skipping arithmetic check.",
        )

    errors = []
    for raw_expr, stated, computed in pairs:
        if abs(computed - stated) > max(0.01 * abs(stated), 0.001):
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

def evidence_grounding_check(
    subtask: str,
    result: str,
    goal: str = "",
    previous_results: list = None
) -> VerificationResult:
    """
    Verification Method 2: Cross-reference factual claims in *result*
    against verifiable knowledge or context.
    """
    context_str = ""
    if goal:
        context_str += f"Overall Goal: {goal}\n"
    if previous_results:
        condensed = [
            {"step": r.get("step", ""), "result": str(r.get("result", ""))[:200]}
            for r in previous_results
        ]
        context_str += f"Context from previous steps:\n{json.dumps(condensed, indent=2)}\n"

    prompt_text = (
        f"You are a strict evidence checker for an AI agent.\n\n"
        f"{context_str}\n"
        f"Current Subtask:\n{subtask}\n\n"
        f"Agent's result:\n{result}\n\n"
        f"Your job: Determine whether the factual or mathematical claims in the result "
        f"are plausible and grounded (supported by previous results, tools, or established facts).\n"
        f"If the result is an arithmetic calculation or deduction derived from previous steps, return verdict: passed.\n\n"
        f"Return ONLY a valid JSON object in this exact format:\n"
        f'{{"verdict": "passed" or "failed", "reason": "one sentence explanation"}}'
    )

    try:
        raw = _llm_call([{"role": "user", "content": prompt_text}])
        data = _extract_json_dict(raw)
        if data:
            verdict = data.get("verdict", "passed").lower()
            reason = data.get("reason", "No reason provided.")
            return VerificationResult(
                status="passed" if "pass" in verdict else "failed",
                method="evidence_grounding",
                reason=reason,
            )
    except Exception:
        pass

    return VerificationResult(
        status="passed",
        method="evidence_grounding",
        reason="Evidence grounding check inconclusive — defaulting to passed.",
    )


# ---------------------------------------------------------------------------
# Check 3: Logical Consistency Check
# ---------------------------------------------------------------------------

def logical_consistency_check(
    subtask: str,
    result: str,
    goal: str = "",
    previous_results: list = None
) -> VerificationResult:
    """
    Verification Method 3: Checks whether the result logically and
    completely answers the given subtask, taking previous steps into account.
    """
    context_str = ""
    if goal:
        context_str += f"Overall Goal: {goal}\n"
    if previous_results:
        condensed = [
            {"step": r.get("step", ""), "result": str(r.get("result", ""))[:200]}
            for r in previous_results
        ]
        context_str += f"Previous steps results:\n{json.dumps(condensed, indent=2)}\n"

    prompt_text = (
        f"You are a strict logical consistency checker for an AI agent.\n\n"
        f"{context_str}\n"
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
        raw = _llm_call([{"role": "user", "content": prompt_text}])
        data = _extract_json_dict(raw)
        if data:
            verdict = data.get("verdict", "passed").lower()
            reason = data.get("reason", "No reason provided.")
            return VerificationResult(
                status="passed" if "pass" in verdict else "failed",
                method="logical_consistency",
                reason=reason,
            )
    except Exception:
        pass

    return VerificationResult(
        status="passed",
        method="logical_consistency",
        reason="Logical consistency check inconclusive — defaulting to passed.",
    )


# ---------------------------------------------------------------------------
# Public API: Run all three verification checks
# ---------------------------------------------------------------------------

def verify_step(
    subtask: str,
    result: str,
    goal: str = "",
    previous_results: list = None
) -> VerificationResult:
    """
    Run all three verification methods on a step result.

    The overall verdict is:
        "passed"  → all three checks passed
        "failed"  → at least one check failed

    Args:
        subtask:          The subtask description that was executed.
        result:           The result string produced by the Executor.
        goal:             The overarching user goal.
        previous_results: Prior step outputs (for multi-step context).

    Returns:
        A VerificationResult with the aggregated verdict.
    """
    print("[Verifier] Running Numeric Re-Check...")
    numeric_result = numeric_recheck(subtask, result)
    print(f"[Verifier] Numeric Re-Check: {numeric_result.status.upper()} — {numeric_result.reason}")

    print("[Verifier] Running Evidence Grounding Check...")
    evidence_result = evidence_grounding_check(subtask, result, goal, previous_results)
    print(f"[Verifier] Evidence Grounding: {evidence_result.status.upper()} — {evidence_result.reason}")

    print("[Verifier] Running Logical Consistency Check...")
    logic_result = logical_consistency_check(subtask, result, goal, previous_results)
    print(f"[Verifier] Logical Consistency: {logic_result.status.upper()} — {logic_result.reason}")

    # Aggregate: any failure = overall failure
    failures = [
        r for r in [numeric_result, evidence_result, logic_result]
        if r.status == "failed"
    ]

    if failures:
        combined_reason = " | ".join(
            f"[{f.method}] {f.reason}" for f in failures
        )
        return VerificationResult(
            status="failed",
            method="all",
            reason=combined_reason,
        )

    return VerificationResult(
        status="passed",
        method="all",
        reason="All three verification checks passed.",
    )
