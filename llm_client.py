"""
LLM client with automatic fallback chain.
Order: Qwen (Bitget credit) -> Groq -> OpenRouter
All API keys are read from environment variables (GitHub Secrets
when running via Actions, or a local .env file for testing).
"""

import os
import requests
from datetime import datetime, timezone

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
        "model": "qwen3-32b",
    },
    {
        "name": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "model": "qwen/qwen3.6-plus:free",
    },
]


def _call_provider(provider: dict, prompt: str, timeout: int = 30) -> dict:
    """Call a single provider. Returns dict {success, content, error}."""
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
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        if resp.status_code == 429:
            return {"success": False, "error": "rate_limit_exceeded"}
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return {"success": True, "content": content}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}


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
