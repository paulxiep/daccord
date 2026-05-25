"""Shared RPM throttle + transient-error retry for API clients in the eval harness.

Two concerns, kept in one tiny module because both are about *pacing the eval
runner against flaky free-tier APIs*:

1. **`api_throttle`** — one sliding-window limiter every remote-API client
   shares, paced at the strictest per-provider RPM (Gemini + Cerebras: 5 RPM).
   Calls fire in tandem at one global rate; wall-time is predictable
   (`call_count / RPM`). `daccord.costs.preflight` only enforces RPD, so back-
   to-back bursts would otherwise exhaust the per-minute ceiling mid-run.

2. **`gemini_retry_on_transient`** — Gemini's `generativelanguage` backend
   intermittently returns 503 ("high demand") and 429 (over-quota retries
   under the SDK's tenacity layer). Per-call retry with exponential backoff
   means one transient hiccup doesn't burn the whole baseline run.

Local-only clients (RetrievalClient, LocalHFClient) bypass both — they don't
hit a remote rate-limited surface.

Single-process / single-threaded runner today; the deque is not lock-guarded.
If the runner ever goes multi-threaded, wrap calls in a `threading.Lock`.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

# Per-provider free-tier RPM ceilings (verified 2026-05-25):
# - Gemini 3.1 Flash Lite: 15 RPM (https://ai.google.dev/gemini-api/docs/rate-limits)
# - Groq Llama 3.3 / 4 Scout / Qwen 3-32B: ≥30 RPM per model
# - Cerebras: 5 RPM (https://inference-docs.cerebras.ai/support/rate-limits) — not currently called
#
# 10 RPM (~6-second spacing) is safe under Gemini's 15 RPM ceiling with
# headroom for clock skew + SDK retry overshoot. With pair-major iteration
# in `runner.py`, each pair triggers 4 generators back-to-back inside ONE
# collective cooldown cycle — so the effective per-pair wait is ~1/4 the
# old generator-major wait, even without bumping RPM further. Per-provider
# rates stay well under their caps because each pair contributes at most
# 1 call to each provider.
_API_RPM = 10
_CALL_TIMES: deque[float] = deque()


def api_throttle() -> None:
    """Block until another remote API call fits inside the rolling 60s window.

    All eval-runner API clients call this *after* `preflight` and *before*
    issuing the SDK call. See module docstring for rationale.
    """
    while True:
        now = time.monotonic()
        while _CALL_TIMES and now - _CALL_TIMES[0] > 60.0:
            _CALL_TIMES.popleft()
        if len(_CALL_TIMES) < _API_RPM:
            _CALL_TIMES.append(now)
            return
        # Sleep until the oldest entry falls out of the window, plus a small
        # safety margin to avoid hammering the boundary.
        wait = 60.0 - (now - _CALL_TIMES[0]) + 0.5
        time.sleep(max(0.5, wait))


def gemini_retry_on_transient[T](fn: Callable[[], T]) -> T:
    """Run `fn` and retry on Gemini transient errors (5xx, 429) with backoff.

    Sleeps 30s, 60s, 120s between attempts (max 4 attempts → ~3.5min worst
    case). Non-transient errors (400/401/403, parse failures) propagate
    immediately. Lazy-imports `google.genai.errors` so non-Gemini callers
    don't pay the import cost.
    """
    from google.genai.errors import (  # type: ignore[import-not-found]
        ClientError,
        ServerError,
    )

    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            return fn()
        except ServerError as exc:
            last_exc = exc
        except ClientError as exc:
            if getattr(exc, "code", None) != 429:
                raise
            last_exc = exc
        if attempt < 3:
            time.sleep(30.0 * (2**attempt))
    assert last_exc is not None  # for type-checker; only reachable after retries
    raise last_exc
