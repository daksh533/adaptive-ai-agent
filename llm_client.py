"""
llm_client.py

Single shared LLM gateway for the whole agent.

Supports both Groq and Google Gemini (OpenAI-compatible) with:
  • Provider selection via LLM_PROVIDER=groq or LLM_PROVIDER=gemini
  • Automatic fallback between Groq and Gemini if daily quota is reached
  • Centralized retry loop with exponential backoff on 429 rate limits
  • Request throttling to prevent rate limit spikes

Environment variables (loaded from ``.env``):
    LLM_PROVIDER   – "gemini" | "groq" | "auto" (default: "auto")
    GOOGLE_API_KEY – Google Gemini API key (or GEMINI_API_KEY)
    GEMINI_MODEL   – optional, defaults to "gemini-1.5-flash"
    GROQ_API_KEY   – Groq API key
    GROQ_MODEL     – optional, defaults to "llama-3.1-8b-instant"
"""

from __future__ import annotations

import json
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

# ---------------------------------------------------------------------------
# Provider & Model Settings
# ---------------------------------------------------------------------------

DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
DEFAULT_GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
DEFAULT_GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

_MIN_CALL_INTERVAL = float(os.getenv("LLM_CALL_INTERVAL", "1.5"))
_last_call_time: float = 0.0

_active_client: OpenAI | None = None
_active_provider: str | None = None
_active_model: str | None = None


def _throttle() -> None:
    """Enforce minimum gap between calls to prevent rate limit spikes."""
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < _MIN_CALL_INTERVAL:
        time.sleep(_MIN_CALL_INTERVAL - elapsed)
    _last_call_time = time.time()


def detect_provider() -> str:
    """
    Determine which provider to use.
    If explicitly set via LLM_PROVIDER, honor it.
    Otherwise prefer Gemini if GOOGLE_API_KEY/GEMINI_API_KEY is available,
    falling back to Groq.
    """
    requested = os.getenv("LLM_PROVIDER", "").strip().lower()
    if requested in ("gemini", "google"):
        return "gemini"
    if requested == "groq":
        return "groq"

    # Auto-detection: prefer Gemini if key is provided (higher free quota)
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("GROQ_API_KEY"):
        return "groq"

    return "groq"


def get_client(provider: str | None = None) -> tuple[OpenAI, str, str]:
    """
    Return (client, provider_name, default_model).
    Initialises client lazily.
    """
    global _active_client, _active_provider, _active_model

    prov = provider or detect_provider()

    if _active_client is not None and _active_provider == prov:
        return _active_client, _active_provider, _active_model

    if prov == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY / GEMINI_API_KEY is not set. Add it to .env:\n"
                "    GOOGLE_API_KEY=your_key_here"
            )
        _active_provider = "gemini"
        _active_model = DEFAULT_GEMINI_MODEL
        _active_client = OpenAI(
            api_key=api_key,
            base_url=DEFAULT_GEMINI_BASE_URL,
            timeout=60.0,
            max_retries=0,
        )
        print(f"[LLM] Active Provider: Google Gemini (model: {_active_model})")
        return _active_client, _active_provider, _active_model

    # Default to Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to .env:\n"
            "    GROQ_API_KEY=your_key_here"
        )
    _active_provider = "groq"
    _active_model = DEFAULT_GROQ_MODEL
    _active_client = OpenAI(
        api_key=api_key,
        base_url=DEFAULT_GROQ_BASE_URL,
        timeout=60.0,
        max_retries=0,
    )
    print(f"[LLM] Active Provider: Groq (model: {_active_model})")
    return _active_client, _active_provider, _active_model


# Backwards compatibility
DEFAULT_MODEL = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
DEFAULT_BASE_URL = DEFAULT_GROQ_BASE_URL


# ---------------------------------------------------------------------------
# Retry-aware chat completion with fallback
# ---------------------------------------------------------------------------

def _retry_delay(exc: Exception, attempt: int) -> float:
    """Honour the delay suggested by API when present, else back off."""
    wait = 3.0 * (1.5 ** attempt)
    match = re.search(
        r"(?:try again in|retry after|in)\s+(?:(?P<minutes>\d+)m)?\s*(?:(?P<millis>\d+)ms|(?P<seconds>[\d\.]+)s)",
        str(exc),
        re.IGNORECASE,
    )
    if match:
        if match.group("millis"):
            wait = max(float(match.group("millis")) / 1000.0 + 0.5, 2.0)
        else:
            minutes = float(match.group("minutes") or 0)
            seconds = float(match.group("seconds") or 0)
            wait = max(minutes * 60.0 + seconds + 1.0, 2.0)
    return wait


MAX_SINGLE_WAIT = float(os.getenv("LLM_MAX_RETRY_WAIT", "120"))


def _is_daily_quota_error(exc: Exception) -> bool:
    """Detect non-recoverable 'tokens per day' or daily quota limits."""
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("tokens per day", "tpd", "daily limit", "daily quota", "resource_exhausted")
    )


def _try_fallback_provider() -> bool:
    """
    If active provider is Groq and Google API key exists, switch to Gemini.
    If active provider is Gemini and Groq key exists, switch to Groq.
    Returns True if successfully switched.
    """
    global _active_client, _active_provider, _active_model
    current = _active_provider or detect_provider()

    if current == "groq" and (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        print("\n[LLM] ⚠️ Groq quota reached. Automatically switching to Google Gemini...")
        get_client("gemini")
        return True
    elif current == "gemini" and os.getenv("GROQ_API_KEY"):
        print("\n[LLM] ⚠️ Gemini quota reached. Automatically switching to Groq...")
        get_client("groq")
        return True

    return False


def chat_completion(
    messages: list,
    *,
    model: str | None = None,
    tools: list | None = None,
    tool_choice: str | None = None,
    temperature: float | None = None,
    max_retries: int = 5,
    label: str = "LLM",
):
    """
    Call the chat-completions API with retries and automatic multi-provider fallback.
    """
    client, provider_name, default_model = get_client()

    target_model = model or default_model

    kwargs: dict = {"model": target_model, "messages": messages}
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if temperature is not None:
        kwargs["temperature"] = temperature

    for attempt in range(max_retries):
        try:
            _throttle()
            return client.chat.completions.create(**kwargs)

        except RateLimitError as exc:
            if _is_daily_quota_error(exc):
                if _try_fallback_provider():
                    client, _, new_model = get_client()
                    kwargs["model"] = new_model
                    continue
                raise RuntimeError(
                    f"Daily token quota exhausted on {provider_name}. "
                    "Switch providers in .env (LLM_PROVIDER=gemini or LLM_PROVIDER=groq) or wait for reset."
                ) from exc

            wait = _retry_delay(exc, attempt)
            if wait > MAX_SINGLE_WAIT:
                if _try_fallback_provider():
                    client, _, new_model = get_client()
                    kwargs["model"] = new_model
                    continue
                raise RuntimeError(f"Rate limit wait of {wait:.0f}s is too long.") from exc

            print(f"[{label}] Rate limit hit on {provider_name}. Pausing {wait:.1f}s before retrying (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)

        except Exception as exc:
            err_str = str(exc).lower()
            if "not found" in err_str or "404" in err_str or "does not exist" in err_str:
                if provider_name == "gemini" and kwargs.get("model") != "gemini-1.5-flash":
                    fallback_model = "gemini-2.0-flash" if kwargs.get("model") != "gemini-2.0-flash" else "gemini-1.5-flash"
                    print(f"[{label}] Note: Model '{kwargs.get('model')}' not found on endpoint. Retrying with '{fallback_model}'...")
                    kwargs["model"] = fallback_model
                    continue
            if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                if _is_daily_quota_error(exc) or "daily" in err_str:
                    if _try_fallback_provider():
                        client, _, new_model = get_client()
                        kwargs["model"] = new_model
                        continue
                wait = _retry_delay(exc, attempt)
                print(f"[{label}] Rate limit / quota hit on {provider_name}. Pausing {wait:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                # If tool error or parameter issue with Gemini/Groq, raise
                raise

    # Last attempt
    _throttle()
    return client.chat.completions.create(**kwargs)


def message_content(response) -> str:
    """Return the text content of the first choice, stripped."""
    if not response or not response.choices:
        return ""
    return (response.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def extract_json_dict(text: str) -> dict:
    """
    Best-effort extraction of a JSON object from an LLM response.
    Handles plain JSON, fenced ```json blocks, and surrounding prose.
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
