"""
verification/risk_classifier.py

Phase 10 — Adaptive Risk-Based Verification

Classifies each subtask + result into a risk tier so the verifier can
decide which checks to run:

    LOW    -> formatting / summarisation / no external facts, no math
              => numeric_recheck only (0 LLM calls)

    MEDIUM -> factual claims, web / KB lookups, citations required
              => numeric_recheck + evidence_grounding_check (1 LLM call)

    HIGH   -> arithmetic calculations, financial figures, multi-step
              numeric dependencies
              => all three checks (2 LLM calls) — same as Phase 6

Strategy:
  1. Try keyword-based rule classification first (0 LLM calls, instant).
  2. Only call the LLM when the keywords are ambiguous (saves rate-limit quota).
"""

from __future__ import annotations

import re

from llm_client import chat_completion, extract_json_dict, message_content


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RISK_TIERS = {"low", "medium", "high"}

# Keywords that strongly indicate HIGH risk (arithmetic / financial)
_HIGH_KEYWORDS = re.compile(
    r"\b("
    r"calculate|calculation|calculator|computing|compute|arithmetic|percent|percentage"
    r"|math|sum|total|add|addition|subtract|subtraction|multiply|multiplication|divide|division"
    r"|financial|profit|loss|revenue|cost|budget|price|amount|figure|equation"
    r"|=\s*\d|result is \d|\d\s*\+\s*\d|\d\s*-\s*\d|\d\s*\*\s*\d|\d\s*/\s*\d"
    r")\b|%",
    re.IGNORECASE,
)

# Keywords that strongly indicate MEDIUM risk (lookup / factual / retrieval)
_MEDIUM_KEYWORDS = re.compile(
    r"\b("
    r"search|look\s*up|find|fetch|retrieve|web|url|source|cite|citation"
    r"|according to|based on|research|fact|data|population|capital|country"
    r"|founded|invented|discovered|written by|author|published"
    r")\b",
    re.IGNORECASE,
)

# Keywords that strongly indicate LOW risk (formatting / summarisation / code / programming / general)
_LOW_KEYWORDS = re.compile(
    r"\b("
    r"format|summariz|summarise|rewrite|list|bullet|paragraph|introduc"
    r"|present|organiz|restructur|rephrase|section|heading|title|display"
    r"|code|script|python|program|function|def\b|class\b|loop|import\b"
    r"|fibonacci|print|algorithm|string|sort\b|array|explain|generate"
    r")\b",
    re.IGNORECASE,
)

_SYSTEM_PROMPT = (
    "You are a risk assessor for an AI agent verification system.\n"
    "Classify the verification risk level of a subtask result.\n\n"
    "LOW    - purely textual / formatting / summarisation. No math, no external facts.\n"
    "MEDIUM - factual claims needing web or knowledge-base grounding.\n"
    "HIGH   - arithmetic, financial figures, or multi-step numeric calculations.\n\n"
    "Return ONLY valid JSON: "
    "{\"risk_level\": \"low\"|\"medium\"|\"high\", \"reason\": \"one sentence\"}"
)


# ---------------------------------------------------------------------------
# Rule-based fast classifier (no LLM)
# ---------------------------------------------------------------------------

def _rule_classify(subtask: str, result: str) -> str | None:
    """
    Try to determine risk tier from keywords alone.
    Returns "low" / "medium" / "high", avoiding expensive LLM calls.
    """
    combined = subtask + " " + result

    high_score   = len(_HIGH_KEYWORDS.findall(combined))
    medium_score = len(_MEDIUM_KEYWORDS.findall(combined))
    low_score    = len(_LOW_KEYWORDS.findall(combined))

    # Clear winner
    if high_score > 0 and high_score >= medium_score and high_score >= low_score:
        return "high"
    if medium_score > 0 and medium_score > low_score:
        return "medium"
    if low_score > 0 and high_score == 0 and medium_score == 0:
        return "low"

    # If neither arithmetic nor search keywords exist, it is a safe standard/code task (LOW risk)
    if high_score == 0 and medium_score == 0:
        return "low"

    return "high" if high_score > 0 else "medium"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_risk(subtask: str, result: str) -> dict:
    """
    Classify the risk level of a subtask result.

    First attempts a fast keyword-based rule check (no LLM call).
    Falls back to an LLM call only when the rules are ambiguous.
    On any failure, conservatively returns ``"high"``.

    Args:
        subtask: The subtask description string.
        result:  The agent's result string for that subtask.

    Returns:
        dict with keys:
            risk_level : str  ("low" | "medium" | "high")
            reason     : str  (one-sentence explanation)
    """
    # ── Step 1: fast rule-based check (free, instant) ─────────────────────
    rule_tier = _rule_classify(subtask, result)
    if rule_tier is not None:
        reason_map = {
            "low":    "Keyword rules: formatting, coding, or standard non-arithmetic task.",
            "medium": "Keyword rules: factual lookup or retrieval required.",
            "high":   "Keyword rules: arithmetic or numeric calculation detected.",
        }
        print(f"[RiskClassifier] {rule_tier.upper()} (rules) - {reason_map[rule_tier]}")
        return {"risk_level": rule_tier, "reason": reason_map[rule_tier]}

    # ── Step 2: LLM fallback for ambiguous cases ──────────────────────────
    user_content = (
        f"Subtask:\n{subtask}\n\n"
        f"Result (first 300 chars):\n{result[:300]}\n\n"
        "Classify the risk level."
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    try:
        response = chat_completion(messages, temperature=0.0, label="RiskClassifier")
        raw = message_content(response)
        data = extract_json_dict(raw)
        if data:
            level = str(data.get("risk_level", "high")).strip().lower()
            reason = str(data.get("reason", "")).strip()
            if level not in RISK_TIERS:
                level = "high"
                reason = "Unknown tier from LLM; defaulting to high."
            print(f"[RiskClassifier] {level.upper()} (llm) - {reason}")
            return {"risk_level": level, "reason": reason}
    except Exception as exc:  # noqa: BLE001
        print(f"[RiskClassifier] LLM classification failed ({exc}). Defaulting to HIGH.")

    fallback = {
        "risk_level": "high",
        "reason": "Classification failed — defaulting to high (conservative fail-safe).",
    }
    print(f"[RiskClassifier] HIGH - {fallback['reason']}")
    return fallback
