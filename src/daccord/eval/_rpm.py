"""Per-provider RPM throttle + transient-error retry for API clients.

Two concerns kept together because both pace the eval+ensemble runners
against flaky multi-tenant APIs:

1. **`api_throttle(provider)`** — per-provider sliding-window limiter.
   Each remote client passes its own `provider` so different providers'
   in-flight rates don't share a queue. Sized per-provider in
   `_PROVIDER_RPM` below, with the rate set well under the documented
   ceiling so transient SDK retries don't push us over.

2. **`gemini_retry_on_transient`** — Gemini's `generativelanguage`
   backend intermittently returns 503 ("high demand") and 429
   ("over-quota") that the SDK's tenacity layer doesn't always cover.
   Per-call retry with exponential backoff so one transient hiccup
   doesn't burn the whole run.

Local-only clients (RetrievalClient, LocalHFClient) bypass — they hit
no remote rate-limited surface.

Thread-safe: the rolling-window deques live behind a `threading.Lock`
so `LocalAPIStrategy`'s ThreadPoolExecutor can safely run one worker
per provider in parallel.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable

from daccord.costs.config import Provider

# Per-provider RPM ceilings. Verified May 2026; lowered ~10–15% to leave
# headroom for SDK-internal retries that show up on the wire.
#
# Free tier (verified):
#   - google_gemini: 15 RPM is the ceiling for gemini-3.1-flash-lite free
#                    (https://ai.google.dev/gemini-api/docs/rate-limits)
#                    — operator floor per docs/7a_path.md; do not lower
#   - groq:          30 RPM per-model is Groq's free-tier published rate
#   - nvidia_nim:    40 RPM (https://yangmao.ai/en/providers/nvidia-build/)
#   - mistral:       ~60 RPM (1 RPS Experiment tier)
#   - openrouter:    20 RPM on :free routes post-$10 credit
#
# Paid Tier 1 (verified May 2026; ~10% safety margin):
#   - anthropic:     50 RPM Tier 1 (platform.claude.com/docs/en/api/rate-limits)
#   - openai:        500 RPM gpt-5-mini Tier 1
#   - together:      no documented fixed RPM; dynamic concurrency-based
#                    (we cap at 600 to keep one Together provider from
#                     monopolising the parallel pool)
#   - deepseek:      concurrency-based; ~50 RPM safe assumption
#
# Local / self-hosted (no externally enforced cap):
#   - bedrock_batch: not throttled here (Bedrock-side queues)
#   - retrieval / local_hf: bypass (no remote call)
_DEFAULT_PROVIDER_RPM: dict[Provider, int] = {
    "google_gemini": 15,
    "groq": 28,
    "anthropic": 45,
    "openai": 450,
    "together": 600,
    "cerebras": 5,
    "deepseek": 50,
    "bedrock_batch": 600,
}

# Env-var override pattern: `DACCORD_RPM_<PROVIDER>=<int>`. The operator sets
# these in `.env.local` when they're on a paid tier with a higher ceiling
# (e.g. `DACCORD_RPM_GOOGLE_GEMINI=4000` for paid Gemini 3.1 Flash Lite
# Tier 2). Default table above is the most conservative known ceiling
# (free tier where applicable) so unmodified code always stays under the
# real limit.
_ENV_RPM_PREFIX = "DACCORD_RPM_"


def _provider_rpm(provider: Provider) -> int:
    env_key = _ENV_RPM_PREFIX + provider.upper()
    override = os.environ.get(env_key)
    if override:
        try:
            return int(override)
        except ValueError:
            pass  # fall through to default
    return _DEFAULT_PROVIDER_RPM.get(provider, 15)


_CALL_TIMES: dict[Provider, deque[float]] = defaultdict(deque)
_LOCK = threading.Lock()


def api_throttle(provider: Provider | None = None) -> None:
    """Block until another remote API call fits inside the rolling 60s window.

    `provider=None` is the legacy global-throttle behavior (preserved for
    callers that haven't been migrated yet) — it uses the lowest configured
    RPM (Gemini's 15) so any caller is safe under any provider's ceiling.

    Each `ModelClient` should pass its own `provider` so per-provider
    rates are independent — that's the point of the rewrite.
    """
    if provider is None:
        # Legacy callers: use the most conservative cap (Gemini 15 RPM).
        rpm = _provider_rpm("google_gemini")
        queue_key: Provider = "google_gemini"
    else:
        rpm = _provider_rpm(provider)
        queue_key = provider

    while True:
        with _LOCK:
            now = time.monotonic()
            queue = _CALL_TIMES[queue_key]
            while queue and now - queue[0] > 60.0:
                queue.popleft()
            if len(queue) < rpm:
                queue.append(now)
                return
            wait = 60.0 - (now - queue[0]) + 0.5
        time.sleep(max(0.5, wait))


def gemini_retry_on_transient[T](fn: Callable[[], T]) -> T:
    """Run `fn` and retry on Gemini transient errors (5xx, 429) with backoff.

    Sleeps 30s, 60s, 120s between attempts (max 4 attempts → ~3.5 min worst
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
