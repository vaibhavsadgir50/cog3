"""
Minimal Ollama HTTP client (stdlib only): chat completions for data generation.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Any


def _is_timeout_exc(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, socket.timeout):
        return True
    msg = str(exc).lower()
    return "timed out" in msg or "timeout" in msg


def ollama_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_s: float = 600.0,
) -> str:
    """
    POST /api/chat with stream=false; returns assistant message content.

    Raises urllib.error.URLError on connection errors, ValueError on bad JSON.
    """
    url = base_url.rstrip("/") + "/api/chat"
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        if e.reason is not None and _is_timeout_exc(e.reason):
            raise TimeoutError(
                f"Ollama request exceeded timeout_s={timeout_s!r} "
                f"(try env OLLAMA_TIMEOUT, CLI --timeout, or smaller --per-call)"
            ) from e
        raise
    except socket.timeout as e:
        raise TimeoutError(
            f"Ollama socket timed out after {timeout_s}s "
            f"(increase timeout or reduce transitions per request)"
        ) from e
    data: dict[str, Any] = json.loads(raw)
    msg = data.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str):
        raise ValueError(f"Unexpected Ollama response: {data!r}")
    return content
