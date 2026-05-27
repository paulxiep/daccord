"""Path 2 — paid direct API `PaidAPIStrategy` implementation.

Calls 4 `ModelClient` instances in parallel (one ThreadPoolExecutor worker
per client) for each (framework_pair, seat). Per-call writes via
`append_candidate` from `daccord.ensemble.strategy` make the run
resumable — crashes lose at most one in-flight call.

Default lineup (label-generation-worthy 2025+ small variants, four families):

  - `AnthropicClient(model="claude-haiku-4-5")` — $1/$5, 50 RPM T1
  - `OpenAIClient(model="gpt-5-mini")`           — $0.25/$2, 500 RPM T1
  - `GeminiClient(model="gemini-3.1-flash-lite")` — $0.25/$1.50,
                                                    4000 RPM operator paid
  - `TogetherClient(model="Qwen/Qwen3-235B-A22B-Instruct-2507-tput")`
                                                  — $0.20/$0.60, serverless

See [docs/7a_path.md] §"Path 2" for the rationale + cost + wall-clock.

## Resilience guarantees

Each per-seat worker uses these properties from `daccord.ensemble.strategy`:

  1. **Resume**: at start, `load_completed_source_ids(out_path)` returns the
     set of `source_id`s already on disk; the worker filters those out so a
     re-invocation only calls the model for unfinished prompts.
  2. **Per-call durability**: `append_candidate(out_path, candidate)` does
     an `O_APPEND` write + `fsync` so the row is on disk before the next
     call begins. Crashes after `fsync` cannot lose the row.
  3. **Per-call error containment**: any exception from `client.generate`
     becomes a `parse_error`-flagged `EnsembleCandidate` (via
     `make_error_candidate`), appended to disk, and the worker continues.
     The source_id is recorded as attempted-and-failed; the next resume
     won't re-call it unless the operator deletes the row from disk.
  4. **Cross-seat independence**: each seat has its own output file and
     its own ThreadPoolExecutor worker — one seat crashing doesn't affect
     the other three.

Default behavior is per-pair, sequentially per seat (one ThreadPoolExecutor
worker per ModelClient). For Scope B 2.16K calls per seat, this means
Anthropic Haiku 4.5 at 50 RPM takes 43 min; the other three seats complete
in 4–10 min and idle until Anthropic finishes.

## What this module is NOT

- Not a CLI (that's `scripts/run_ensemble.py`).
- Not a pricing or rate-limit table (those are
  [costs/config.toml](../../../../costs/config.toml) and
  [src/daccord/eval/_rpm.py](../../../eval/_rpm.py)).
- Not a strategy registry (the CLI picks the strategy by `--strategy`
  flag and constructs it inline).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from daccord.ensemble.prompt import BatchPrompt
from daccord.ensemble.schema import EnsembleCandidate
from daccord.ensemble.strategy import (
    RunResult,
    append_candidate,
    load_completed_source_ids,
    make_error_candidate,
    output_path_for,
    prune_parse_errors,
)
from daccord.eval.clients import ModelClient
from daccord.eval.schema import PromptMessages
from daccord.validation import validated

log = logging.getLogger("daccord.ensemble.strategies.paid_api")


def _default_clients() -> list[ModelClient]:
    """Construct the default Path-2 paid-API ensemble (4 seats, 2025+ gen).

    Lazy-imports the client classes so this module is importable in tests
    that don't have all 4 SDK keys set in the environment. Each client's
    `__init__` raises `RuntimeError` if its API key env var is unset.
    """
    from daccord.eval.clients import (
        AnthropicClient,
        GeminiClient,
        OpenAIClient,
        TogetherClient,
    )

    return [
        AnthropicClient(model="claude-haiku-4-5"),
        OpenAIClient(model="gpt-5-mini"),
        GeminiClient(model="gemini-3.1-flash-lite"),
        TogetherClient(model="Qwen/Qwen3-235B-A22B-Instruct-2507-tput"),
    ]


class PaidAPIStrategy:
    """Path 2 — paid direct API `EnsembleStrategy`.

    Construct with an explicit list of `ModelClient` instances OR with no
    arguments to get the default 4-seat lineup. Mid-run swaps (e.g.
    substitute the Together seat for a DeepSeek seat when budget pressure
    bites) happen at construction time — no per-call dispatch overhead.

    `run_pair(framework_pair, prompts, out_dir, *, smoke)` fans the
    prompts across seats in parallel, writes per-seat JSONL outputs
    incrementally, and returns one `RunResult` per seat.

    `smoke` is accepted to match the `EnsembleStrategy` protocol but is
    not used here — Path 2 has no separate smoke-mode S3 prefixing
    (output files always land under `out_dir`). Caller passes a different
    `out_dir` for smoke vs full runs if needed.
    """

    name = "local-api-paid"

    @validated
    def __init__(self, clients: Sequence[ModelClient] | None = None) -> None:
        # Sequence (covariant) rather than list (invariant) on the parameter
        # so tests can pass `list[_FakeModelClient]` without a cast. Internal
        # storage stays as list for stable iteration order.
        self._clients: list[ModelClient] = (
            list(clients) if clients is not None else _default_clients()
        )
        if not self._clients:
            raise ValueError("PaidAPIStrategy needs at least one ModelClient")
        # Detect duplicate model IDs upfront — two clients writing to the
        # same output file would race + corrupt the JSONL.
        seen: set[str] = set()
        for c in self._clients:
            if c.model in seen:
                raise ValueError(f"duplicate model id in clients list: {c.model!r}")
            seen.add(c.model)

    @property
    def models(self) -> list[str]:
        return [c.model for c in self._clients]

    def run_pair(
        self,
        framework_pair: str,
        prompts: list[BatchPrompt],
        out_dir: Path,
        *,
        smoke: bool,
        retry_errors: bool = False,
    ) -> dict[str, RunResult]:
        """Run one framework-pair across all seats. See class docstring.

        `retry_errors=True` scrubs parse_error rows from each seat's
        output JSONL before computing the resume set, so any prior
        per-(seat, source_id) failure gets re-called. Successful rows
        from other seats — and successful rows on the same seat for
        other source_ids — stay intact (no work duplication).
        """
        _ = smoke  # see class docstring

        # Per-seat pruning runs BEFORE the parallel ThreadPoolExecutor so
        # the resume-set computation in each worker sees the cleaned file.
        # Each seat's file is independent: pruning seat A's parse_errors
        # does not touch seats B/C/D's files.
        if retry_errors:
            for client in self._clients:
                seat_path = output_path_for(out_dir, framework_pair, client.model)
                prune_parse_errors(seat_path)

        results: dict[str, RunResult] = {}
        # One worker per seat — distinct providers run concurrently, but
        # the per-provider RPM throttle inside `api_throttle(provider)`
        # serializes calls to the same provider when (e.g.) two clients
        # share a provider. For the default 4-seat lineup providers are
        # already disjoint.
        with ThreadPoolExecutor(max_workers=len(self._clients)) as pool:
            futures = {
                pool.submit(
                    self._run_one_client,
                    client=client,
                    framework_pair=framework_pair,
                    prompts=prompts,
                    out_dir=out_dir,
                ): client.model
                for client in self._clients
            }
            for fut in as_completed(futures):
                model = futures[fut]
                try:
                    results[model] = fut.result()
                except Exception as exc:
                    # Per-worker exceptions outside `run_one_client`'s own
                    # try/except are unexpected. Don't kill the run — log
                    # and continue with the other seats.
                    log.error(
                        "[paid-api] worker %s / %s crashed unexpectedly: %s",
                        framework_pair,
                        model,
                        exc,
                    )
        return results

    @validated
    def _run_one_client(
        self,
        *,
        client: ModelClient,
        framework_pair: str,
        prompts: list[BatchPrompt],
        out_dir: Path,
    ) -> RunResult:
        start = time.monotonic()
        out_path = output_path_for(out_dir, framework_pair, client.model)

        # Resume support: skip prompts already on disk.
        completed = load_completed_source_ids(out_path)
        if completed:
            log.info(
                "[paid-api] %s / %s: resuming, %d source_ids already on disk",
                framework_pair,
                client.model,
                len(completed),
            )
        remaining = [p for p in prompts if p.source_id not in completed]

        parse_ok = 0
        parse_errors = 0
        for prompt in remaining:
            candidate = self._call_one_prompt(client, framework_pair, prompt)
            try:
                append_candidate(out_path, candidate)
            except Exception as exc:
                # Disk-write failures are catastrophic for resume — log and
                # propagate. The next resume picks up correctly because
                # this candidate's source_id was NOT persisted.
                log.error(
                    "[paid-api] disk write failed for %s / %s / %s: %s",
                    framework_pair,
                    client.model,
                    prompt.source_id,
                    exc,
                )
                raise
            if candidate.parse_error is None:
                parse_ok += 1
            else:
                parse_errors += 1

        elapsed = time.monotonic() - start
        log.info(
            "[paid-api] %s / %s done: processed=%d ok=%d errors=%d resumed=%d (%.1fs)",
            framework_pair,
            client.model,
            len(remaining),
            parse_ok,
            parse_errors,
            len(completed),
            elapsed,
        )
        return RunResult(
            framework_pair=framework_pair,
            model=client.model,
            output_path=str(out_path),
            total_processed=len(remaining),
            parse_ok=parse_ok,
            parse_errors=parse_errors,
            resumed_from_disk=len(completed),
            seconds_elapsed=elapsed,
        )

    @validated
    def _call_one_prompt(
        self,
        client: ModelClient,
        framework_pair: str,
        prompt: BatchPrompt,
    ) -> EnsembleCandidate:
        """Call one prompt; convert success/failure into an `EnsembleCandidate`.

        Failure modes captured here:
          - `ModelResponse.parse_error` non-None (SDK returned, JSON
            unparseable / schema invalid). Recorded as a parse_error row.
          - Any exception from `client.generate` (timeout, network error,
            unexpected SDK exception). Recorded as a parse_error row with
            the exception type + message.

        The individual `*Client` adapters in `daccord.eval.clients` already
        catch most SDK errors and return `parse_error`-flagged
        `ModelResponse` — this outer except is the belt over the
        suspenders for anything that escapes.
        """
        messages = PromptMessages(system=prompt.system, user=prompt.user)
        run_id = framework_pair
        batch_id = client.model

        try:
            response: Any = client.generate(messages=messages, run_id=run_id, batch_id=batch_id)
        except Exception as exc:
            return make_error_candidate(
                prompt=prompt,
                model=client.model,
                error_message=f"{type(exc).__name__}: {exc}",
            )

        if response.parse_error is not None or response.top1 is None:
            return make_error_candidate(
                prompt=prompt,
                model=client.model,
                error_message=response.parse_error or "model returned no candidate",
            )

        top1 = response.top1
        return EnsembleCandidate(
            source_id=prompt.source_id,
            source_jurisdiction=prompt.source_jurisdiction,
            source_framework=prompt.source_framework,
            source_citation_id=prompt.source_citation_id,
            source_mechanism=prompt.source_mechanism,
            target_jurisdiction=prompt.target_jurisdiction,
            target_framework=prompt.target_framework,
            model=client.model,
            citation_id=top1.citation_id,
            target_mechanism=top1.target_mechanism,
            mapping_justification=top1.mapping_justification,
            parse_error=None,
        )
