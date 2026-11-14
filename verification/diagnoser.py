"""
verification/diagnoser.py

Phase 7 — Structured Error Diagnosis

Classifies verification failures into a 6-class error taxonomy:
  1. calculation_error   — numeric re-check failed
  2. tool_error          — tool (web_search/python/rag) returned an error
  3. grounding_error     — evidence grounding failed
  4. consistency_error   — logical consistency failed
  5. incomplete_error    — result does not fully address the subtask
  6. unknown_error       — could not classify
"""

import re


class Diagnosis:
    def __init__(self, error_class: str, reason: str, confidence: float = 1.0):
        self.error_class = error_class      # one of the 6 classes above
        self.reason = reason                # human-readable explanation
        self.confidence = confidence        # 0.0 – 1.0

    def __repr__(self):
        return f"Diagnosis(error_class={self.error_class!r}, confidence={self.confidence})"

    def to_dict(self):
        return {
            "error_class": self.error_class,
            "reason": self.reason,
            "confidence": self.confidence,
        }


# Keywords that hint at each error class
_CLASS_KEYWORDS = {
    "calculation_error": ["numeric", "calculation", "arithmetic", "math", "number", "compute"],
    "tool_error":        ["tool", "web_search", "search", "python", "run_python", "execution", "timeout", "api"],
    "grounding_error":   ["grounding", "evidence", "source", "rag", "knowledge", "cite", "document"],
    "consistency_error": ["consistency", "logical", "contradict", "inconsistent", "context"],
    "incomplete_error":  ["incomplete", "partial", "missing", "not fully", "does not address"],
}


def diagnose(subtask: str, result: str, verification_reason: str, previous_results: list) -> Diagnosis:
    """
    Classify why a verification step failed.

    Args:
        subtask:          The subtask that was executed.
        result:           The result produced by the executor.
        verification_reason: Human-readable reason from the verifier.
        previous_results: Prior step outputs (for context).

    Returns:
        A Diagnosis object with error_class, reason, and confidence.
    """
    text = f"{subtask} {result} {verification_reason}".lower()

    # Score each class by keyword overlap
    best_class = "unknown_error"
    best_score = 0
    for error_class, keywords in _CLASS_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_score = score
            best_class = error_class

    # Heuristic refinements
    if "error" in result.lower()[:200] and best_class == "unknown_error":
        best_class = "tool_error"
    if len(str(result).strip()) < 10 and best_class == "unknown_error":
        best_class = "incomplete_error"

    reason_map = {
        "calculation_error": "Numeric or calculation result could not be verified.",
        "tool_error":        "Underlying tool call failed or returned an error.",
        "grounding_error":   "Result lacks sufficient evidence or source grounding.",
        "consistency_error": "Result contains internal contradictions or logical issues.",
        "incomplete_error":  "Result does not fully address the subtask requirements.",
        "unknown_error":     "Failure cause could not be classified with confidence.",
    }

    confidence = 0.9 if best_score > 0 else 0.5
    return Diagnosis(
        error_class=best_class,
        reason=reason_map[best_class],
        confidence=confidence,
    )


def diagnose_step(subtask: str, result: str, verification_reason: str, previous_results: list) -> dict:
    """Public API: returns a dict suitable for merging into GraphState."""
    d = diagnose(subtask, result, verification_reason, previous_results)
    return d.to_dict()
