"""
llm_client.py

Single shared LLM gateway for the whole agent.

Every module (planner, executor, verifier, corrector, replanner) used to build
its own ``OpenAI`` client and its own retry loop. This module centralises that
logic so the model, base URL, timeouts and rate-limit policy are configured in
exactly one place.

Environment variables (loaded from ``.env``):
    GROQ_API_KEY   – required
    GROQ_MODEL     – optional, defaults to ``openai/gpt-oss-20b``
    GROQ_BASE_URL  – optional, defaults to ``https://api.groq.com/openai/v1``
"""

from __future__ import annotations

import json
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
DEFAULT_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

_client: OpenAI | None = None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def get_client() -> OpenAI:
    """
    Return the shared OpenAI-compatible client, creating it on first use.

    Creating the client lazily means importing the agent (and running offline
    tests) does not require ``GROQ_API_KEY`` to be set. The error is raised only
    when a real LLM call is attempted.
    """
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to the project .env file:\n"
                "    GROQ_API_KEY=your_key_here"
            )
        _client = OpenAI(
            api_key=api_key,
            base_url=DEFAULT_BASE_URL,
            timeout=60.0,
            max_retries=0,  # retries are handled below
        )
    return _client


# ---------------------------------------------------------------------------
# Retry-aware chat completion
# ---------------------------------------------------------------------------

def _retry_delay(exc: Exception, attempt: int) -> float:
    """Honour the delay suggested by Groq when present, else back off."""
    wait = 2.0 * (1.5 ** attempt)
    # Handles "500ms", "106s" and compound forms like "1m46.27s".
    match = re.search(
        r"try again in (?:(?P<minutes>\d+)m)?(?:(?P<millis>\d+)ms|(?P<seconds>[\d\.]+)s)",
        str(exc),
    )
    if match:
        if match.group("millis"):
            wait = max(float(match.group("millis")) / 1000.0 + 0.5, 1.5)
        else:
            minutes = float(match.group("minutes") or 0)
            seconds = float(match.group("seconds") or 0)
            wait = max(minutes * 60.0 + seconds + 0.5, 1.5)
    return wait


# A demo should fail fast rather than appear frozen. If the API asks us to wait
# longer than this, surface a clear error instead of sleeping.
MAX_SINGLE_WAIT = float(os.getenv("LLM_MAX_RETRY_WAIT", "120"))


def _is_daily_quota_error(exc: Exception) -> bool:
    """Detect the non-recoverable 'tokens per day' free-tier limit."""
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("tokens per day", "tpd", "daily limit", "daily quota")
    )


def _quota_runtime_error(exc: Exception, wait: float | None = None) -> RuntimeError:
    """Build an actionable error for an exhausted or long rate limit."""
    if _is_daily_quota_error(exc):
        detail = "the Groq free-tier daily token quota is exhausted"
    elif wait is not None:
        detail = (
            f"the API asked us to wait {wait:.0f}s, longer than the "
            f"{MAX_SINGLE_WAIT:.0f}s demo-friendly limit"
        )
    else:
        detail = "the API rate limit is still active"

    return RuntimeError(
        f"Groq rate limit: {detail}. Wait for the quota to reset, "
        "or set GROQ_MODEL in .env to a model with more available quota. "
        "Tip: run 'test_offline.py' to demo the agent logic without API calls."
    )


def chat_completion(
    messages: list,
    *,
    model: str | None = None,
    tools: list | None = None,
    tool_choice: str | None = None,
    temperature: float | None = None,
    max_retries: int = 6,
    label: str = "LLM",
):
    """
    Call the chat-completions API, retrying with exponential backoff on 429s.

    Args:
        messages:     OpenAI-format message list.
        model:        Override the default model.
        tools:        Optional tool schemas.
        tool_choice:  Optional tool-choice policy.
        temperature:  Optional sampling temperature.
        max_retries:  Number of retries on rate limiting.
        label:        Short tag used in log messages (e.g. "Planner").

    Returns:
        The raw ``ChatCompletion`` response.
    """
    kwargs: dict = {"model": model or DEFAULT_MODEL, "messages": messages}
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if temperature is not None:
        kwargs["temperature"] = temperature

    for attempt in range(max_retries):
        try:
            return get_client().chat.completions.create(**kwargs)
        except RateLimitError as exc:
            wait = _retry_delay(exc, attempt)
            if _is_daily_quota_error(exc) or wait > MAX_SINGLE_WAIT:
                raise _quota_runtime_error(exc, wait) from exc
            print(f"[{label}] Rate limit hit. Pausing {wait:.1f}s before retrying...")
            time.sleep(wait)
        except Exception as exc:
            # Some SDK/proxy combinations surface 429s as generic APIStatusError.
            if "429" not in str(exc):
                raise
            wait = _retry_delay(exc, attempt)
            if _is_daily_quota_error(exc) or wait > MAX_SINGLE_WAIT:
                raise _quota_runtime_error(exc, wait) from exc
            print(f"[{label}] Rate limit hit. Pausing {wait:.1f}s before retrying...")
            time.sleep(wait)

    try:
        return get_client().chat.completions.create(**kwargs)
    except RateLimitError as exc:
        raise _quota_runtime_error(exc) from exc
    except Exception as exc:
        if "429" not in str(exc):
            raise
        raise _quota_runtime_error(exc) from exc


def message_content(response) -> str:
    """Return the text content of the first choice, stripped."""
    return (response.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def extract_json_dict(text: str) -> dict:
    """
    Best-effort extraction of a JSON object from an LLM response.

    Handles plain JSON, fenced ```json blocks, and surrounding prose.
    Returns ``{}`` when no valid object can be found.
    """
    if not text:
        return {}

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(cleaned[start : end + 1])
            return data if isinstance(data, dict) else {}
        except Exception:
            pass

    return {}


def extract_plan(text: str) -> list[str] | None:
    """
    Parse a ``{"plan": [...]}`` response.

    Returns a non-empty list of strings, or ``None`` if the payload is invalid.
    """
    data = extract_json_dict(text)
    plan = data.get("plan")
    if not isinstance(plan, list):
        return None

    steps = [str(step).strip() for step in plan if str(step).strip()]
    return steps or None
