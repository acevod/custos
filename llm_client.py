"""
LLM client with automatic fallback chain.
Order: Qwen (Bitget credit) -> Groq -> OpenRouter
All API keys are read from environment variables - set them with
`export GROQ_API_KEY=...` etc. before running locally (GitHub Actions
injects them from Secrets automatically).

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
MAX_TOKENS = 800               # cap on each completion

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
        "model": "qwen/qwen3.6-27b",  # qwen3-32b was deprecated by Groq (announced 17 Jun 2026)
    },
    {
        "name": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "model": "openrouter/free",  # auto-router - not pinned to one free model
    },
]


def _call_provider(provider: dict, prompt: str, timeout: int = 30) -> dict:
    """Call a single provider, retrying 429 / 5xx / transient network
    errors on the SAME provider before giving up. Returns dict
    {success, content, error}."""
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
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return {"success": True, "content": content}
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
        except Exception as e:
            # Non-transient error (4xx other than 429, malformed JSON) -
            # retrying won't help, fail straight through.
            return {"success": False, "error": str(e)}

    return {"success": False, "error": last_error}


def call_llm(prompt: str) -> dict:
    """
    Main entry point. Tries each provider in order until one succeeds.
    Returns a full dict (not just the content) so it can be logged -
    lets us see which provider was used / which attempts failed.
    """
    attempts = []

    for provider in PROVIDERS:
        result = _call_provider(provider, prompt)
        attempts.append({
            "provider": provider["name"],
            "status": "success" if result["success"] else "failed",
            "error": result.get("error"),
        })

        if result["success"]:
            return {
                "success": True,
                "content": result["content"],
                "provider_used": provider["name"],
                "attempts": attempts,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    # All providers failed - return a failure record, never silent-fail.
    return {
        "success": False,
        "content": None,
        "provider_used": None,
        "attempts": attempts,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": "all providers failed - will retry next scheduled run",
    }


if __name__ == "__main__":
    import json
    result = call_llm("Reply with just the word OK.")
    print(json.dumps(result, indent=2))
