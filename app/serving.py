"""Phase 3 (P3.3): readiness probe for an OpenAI-compatible endpoint (vLLM).

Hermetic-friendly: only talks to whatever ``base_url`` it is handed, with
short timeouts, and treats every failure (connection refused, bad status,
garbage body) as "not ready yet". Used by ``scripts/vllm_serve.py --probe``
and safe to unit-test against a local stub server.
"""

from __future__ import annotations

import time

import httpx


def is_ready(base_url: str, timeout: float = 3.0) -> bool:
    """True iff ``GET {base_url}/v1/models`` returns 200 with models listed."""
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/v1/models", timeout=timeout)
    except httpx.HTTPError:
        return False
    if resp.status_code != 200:
        return False
    try:
        data = resp.json()
    except ValueError:
        return False
    return isinstance(data, dict) and bool(data.get("data"))


def wait_ready(
    base_url: str, timeout: float = 60.0, interval: float = 1.0
) -> bool:
    """Poll ``is_ready`` until it passes or ``timeout`` seconds elapse."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        if is_ready(base_url, timeout=min(3.0, remaining)):
            return True
        time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
