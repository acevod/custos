"""
LLM client with automatic fallback chain.
Order: Qwen (Bitget credit) -> Groq -> OpenRouter
All API keys are read from environment variables - set them with
`export GROQ_API_KEY=...` etc. before running locally (GitHub Actions
injects them from Secrets automatically).

Audit fixes (round 2):
  - A provider only counts as "successful" if its reply is USABLE: HTTP
    200 alone is not enough. call_llm() takes an optional `validator`
    (main.py passes one that requires a parseable decision). Empty,
    truncated (finish_reason=length) or unparseable replies are recorded
    as "unusable_response" and the NEXT provider is tried.
  - Error strings are reduced to short category codes (http_404,
    timeout, ...) before they reach the logs that get committed to a
    public repo - raw exception text embeds the request URL, which can
    be a secret (QWEN_BITGET_BASE_URL). A redacted, truncated response
    body is printed to the job log for diagnosis instead.

Fixes vs the original version:
  - Retries 429 / 5xx / timeouts on the SAME provider (with short
    backoff) before falling through to the next provider - a single
    transient rate-limit no longer permanently skips a provider for
    the whole run.
  - max_tokens caps each response, so a misbehaving provider can't
    blow up the event log (or the cost) with an enormous completion.
  - The docstring no longer claims .env support that was never
    implemented.
"""

import os
import time

import requests
from datetime import datetime, timezone

MAX_RETRIES_PER_PROVIDER = 2   # attempts per provider before falling through
BACKOFF_BASE_SECONDS = 2
MAX_TOKENS = 1500              # cap on each completion (reasoning models spend tokens thinking first)

# ── Provider config ───────────────────────────────────────────
PROVIDERS = [
    {
        "name": "qwen_bitget",
        "base_url": os.environ.get("QWEN_BITGET_BASE_URL", "https://hackathon.bitgetops.com/v1"),
        "api_key": os.environ.get("QWEN_BITGET_API_KEY"),
        "model": "qwen3.8-max",
    },
    {
        "name": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": os.environ.get("GROQ_API_KEY"),
        # Groq deprecated qwen3-32b (17 Jun 2026) and then qwen3.6-27b for free/developer
        # tier; qwen3.8-27b is the documented successor. The old id returned HTTP 404.
        "model": "qwen/qwen3.8-27b",
        # Instruct (non-thinking) mode per Groq's model card: without this the model
        # spends the token budget "thinking" and can be cut off before it answers.
        "extra_body": {"reasoning_effort": "none"},
    },
    {
        "name": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        # Same model family as the other two providers so the prompt/JSON contract
        # behaves alike across the whole chain. Trade-off: pinned to one free model
        # (the earlier `openrouter/free` auto-router survived models being pulled).
        "model": "qwen/qwen3.8-27b:free",
    },
]


def _redact(text: str, provider: dict) -> str:
    """Strip anything URL-like / key-like from text before it is printed."""
    text = str(text)
    for secret in (provider.get("base_url"), provider.get("api_key")):
        if secret:
            text = text.replace(secret, "[redacted]")
    return text[:300]


def _extract_text(message_content) -> str | None:
    """OpenAI-compatible providers return a string, but some return a
    list of content parts (or None). Normalize to str or None."""
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        parts = []
        for part in message_content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts) if parts else None
    return None


def _call_provider(provider: dict, prompt: str, timeout: int = 30) -> dict:
    """Call a single provider, retrying 429 / 5xx / transient network
    errors on the SAME provider before giving up. Returns dict
    {success, content, finish_reason, error} where `error` is always a
    short category code (never raw exception text - see module docstring)."""
    if not provider["api_key"]:
        return {"success": False, "error": "no_api_key_configured"}

    url = f"{provider['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {provider['api_key']}",
        "Content-Type": "application/json",
    }
    body = {
        "model": provider["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": MAX_TOKENS,
    }
    body.update(provider.get("extra_body") or {})

    retryable = (429, 500, 502, 503, 504)
    last_error = "unknown_error"

    for attempt in range(1, MAX_RETRIES_PER_PROVIDER + 1):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            if resp.status_code in retryable:
                last_error = ("rate_limit_exceeded" if resp.status_code == 429
                              else f"http_{resp.status_code}")
                if attempt < MAX_RETRIES_PER_PROVIDER:
                    time.sleep(BACKOFF_BASE_SECONDS * attempt)
                    continue
                return {"success": False, "error": last_error}
            if resp.status_code >= 400:
                # Non-retryable client error (bad model id, auth, access...).
                # Body goes to the (redacted) job log only, never to data/.
                print(f"[llm:{provider['name']}] HTTP {resp.status_code}: "
                      f"{_redact(resp.text, provider)}")
                return {"success": False, "error": f"http_{resp.status_code}"}
            data = resp.json()
            choice = data["choices"][0]
            content = _extract_text(choice.get("message", {}).get("content"))
            return {"success": True, "content": content,
                    "finish_reason": choice.get("finish_reason")}
        except requests.exceptions.Timeout:
            last_error = "timeout"
            if attempt < MAX_RETRIES_PER_PROVIDER:
                time.sleep(BACKOFF_BASE_SECONDS * attempt)
                continue
            return {"success": False, "error": last_error}
        except requests.exceptions.ConnectionError:
            last_error = "connection_error"
            if attempt < MAX_RETRIES_PER_PROVIDER:
                time.sleep(BACKOFF_BASE_SECONDS * attempt)
                continue
            return {"success": False, "error": last_error}
        except (ValueError, KeyError, IndexError, TypeError):
            # Malformed JSON / unexpected response shape - retrying won't help.
            return {"success": False, "error": "malformed_response"}
        except Exception as e:
            # Category only - str(e) can contain the request URL.
            return {"success": False, "error": f"error_{type(e).__name__}"}

    return {"success": False, "error": last_error}


def call_llm(prompt: str, validator=None) -> dict:
    """
    Main entry point. Tries each provider in order until one returns a
    USABLE reply. `validator(content) -> bool` decides usability; when
    omitted, any non-empty string is accepted. Returns a full dict (not
    just the content) so it can be logged - lets us see which provider
    was used / which attempts failed and why.
    """
    attempts = []

    for provider in PROVIDERS:
        result = _call_provider(provider, prompt)
        status, error = ("success", None) if result["success"] else ("failed", result.get("error"))

        if result["success"]:
            content = result.get("content")
            usable = isinstance(content, str) and content.strip() != ""
            if usable and validator is not None:
                try:
                    usable = bool(validator(content))
                except Exception:
                    usable = False
            if not usable:
                status = "unusable_response"
                error = ("truncated" if result.get("finish_reason") == "length"
                         else "empty_response" if not (isinstance(content, str) and content.strip())
                         else "unparseable_decision")

        attempts.append({"provider": provider["name"], "status": status, "error": error})

        if status == "success":
            return {
                "success": True,
                "content": result["content"],
                "provider_used": provider["name"],
                "attempts": attempts,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    # All providers failed or were unusable - never silent-fail.
    return {
        "success": False,
        "content": None,
        "provider_used": None,
        "attempts": attempts,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": "no provider returned a usable reply - will retry next scheduled run",
    }


if __name__ == "__main__":
    import json
    result = call_llm("Reply with just the word OK.")
    print(json.dumps(result, indent=2))
