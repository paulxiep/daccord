"""End-to-end mock runs for `PaidAPIStrategy` resilience.

Exercises the strategy at realistic Scope-B-per-pair sizes (~30 prompts
per seat) under multiple failure modes:

  - **Intermittent transient errors**: 20% of calls fail with
    `TimeoutError` — strategy should record each as a parse_error row
    and finish with all 30 source_ids on disk.
  - **Mid-run crash + resume**: simulate SIGKILL after ~20 successful
    calls — re-invocation should pick up exactly the remaining ~10.
  - **All-providers parallel**: 4 fake clients running concurrently,
    each writing to its own output file, none stepping on each other.
  - **Idempotent re-run**: re-invoking after a fully-complete run
    should do zero new work.

Each scenario uses the `_FakeModelClient` from `test_ensemble_strategy.py`
but adapted for batch-scale runs.

These tests are the "mock runs" called out in the implementation plan.
They run in <5 seconds (no real SDKs, no real network) — fast enough to
catch resilience regressions in CI.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

from daccord.ensemble import BatchPrompt
from daccord.ensemble.strategies.paid_api import PaidAPIStrategy
from daccord.ensemble.strategy import (
    output_path_for,
    read_candidates_jsonl,
)
from daccord.eval.schema import CitationCandidate, ModelResponse, PromptMessages

# Mirror the test fake from test_ensemble_strategy.py — duplicated rather
# than imported because pytest test modules aren't intended to be each
# other's libraries.


class _FakeModelClient:
    def __init__(
        self,
        *,
        provider: Any = "test_provider",
        model: str = "test/fake",
        responses: dict[str, Any] | None = None,
        crash_after: int | None = None,
        flake_rate: float = 0.0,
        flake_seed: int = 42,
    ) -> None:
        self.provider = provider
        self.model = model
        self._responses = responses or {}
        self._crash_after = crash_after
        self._flake_rate = flake_rate
        self._rng = random.Random(flake_seed)
        self._calls: list[str] = []

    @property
    def call_count(self) -> int:
        return len(self._calls)

    def generate(self, messages: PromptMessages, *, run_id: str, batch_id: str) -> ModelResponse:
        if self._crash_after is not None and len(self._calls) >= self._crash_after:
            raise SystemExit("CRASH")

        source_id = messages.user.split()[-1]
        self._calls.append(source_id)

        configured = self._responses.get(source_id)
        if configured is not None:
            if isinstance(configured, Exception):
                raise configured
            return configured

        if self._flake_rate > 0 and self._rng.random() < self._flake_rate:
            raise TimeoutError(f"simulated network timeout for {source_id}")

        return ModelResponse(
            model=self.model,
            top1=CitationCandidate(
                citation_id=f"sg-{source_id}",
                target_mechanism=f"sg mapped {source_id}",
                mapping_justification="because",
            ),
            raw_text="{}",
            input_tokens=100,
            output_tokens=50,
            latency_ms=10.0,
        )


def _make_prompts(n: int, *, framework_pair: str = "gdpr__pdpa_sg") -> list[BatchPrompt]:
    return [
        BatchPrompt(
            record_id=f"src_{i:03d}",
            source_id=f"src_{i:03d}",
            source_jurisdiction="eu",
            source_framework="gdpr",
            source_citation_id=f"src_{i:03d}",
            source_mechanism=f"mechanism for src_{i:03d}",
            target_jurisdiction="sg",
            target_framework="pdpa_sg",
            system="system",
            user=f"user prompt for src_{i:03d}",
            max_tokens=256,
        )
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1 — Intermittent transient errors.
# ─────────────────────────────────────────────────────────────────────────────


def test_mock_run_with_intermittent_timeouts(tmp_path: Path) -> None:
    """20% TimeoutError rate. All source_ids land on disk; ~6 are parse_errors."""
    prompts = _make_prompts(30)
    client = _FakeModelClient(model="t/flaky", flake_rate=0.20, flake_seed=42)
    strategy = PaidAPIStrategy(clients=[client])
    results = strategy.run_pair("gdpr__pdpa_sg", prompts, tmp_path, smoke=False)

    rr = results["t/flaky"]
    assert rr.total_processed == 30
    # Every source_id is on disk (either successful or parse_error).
    out_path = output_path_for(tmp_path, "gdpr__pdpa_sg", "t/flaky")
    rows = read_candidates_jsonl(out_path)
    assert {c.source_id for c in rows} == {f"src_{i:03d}" for i in range(30)}
    # Some rows are parse_errors (~6 expected at 20% rate, 30 trials).
    error_rows = [c for c in rows if c.parse_error is not None]
    assert 1 <= len(error_rows) <= 15  # generous bounds; seed=42 is deterministic
    for c in error_rows:
        assert "TimeoutError" in (c.parse_error or "")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2 — Mid-run crash + resume.
# ─────────────────────────────────────────────────────────────────────────────


def test_mock_run_survives_mid_run_crash_and_resumes(tmp_path: Path) -> None:
    """Crash after 20 calls, resume, finish all 30."""
    prompts = _make_prompts(30)
    out_path = output_path_for(tmp_path, "gdpr__pdpa_sg", "t/crash30")

    # Run 1: SystemExit at call 20.
    crashing = _FakeModelClient(model="t/crash30", crash_after=20)
    strategy_1 = PaidAPIStrategy(clients=[crashing])
    with pytest.raises(SystemExit):
        strategy_1._run_one_client(  # noqa: SLF001
            client=crashing,
            framework_pair="gdpr__pdpa_sg",
            prompts=prompts,
            out_dir=tmp_path,
        )

    # After crash, 20 rows persisted.
    after_crash = read_candidates_jsonl(out_path)
    assert len(after_crash) == 20
    assert {c.source_id for c in after_crash} == {f"src_{i:03d}" for i in range(20)}

    # Run 2: fresh client, no crash. Resumes from src_020.
    fresh = _FakeModelClient(model="t/crash30")
    strategy_2 = PaidAPIStrategy(clients=[fresh])
    results = strategy_2.run_pair("gdpr__pdpa_sg", prompts, tmp_path, smoke=False)

    rr = results["t/crash30"]
    assert rr.resumed_from_disk == 20
    assert rr.total_processed == 10
    assert rr.parse_ok == 10
    assert fresh.call_count == 10
    assert list(fresh._calls) == [  # noqa: SLF001
        f"src_{i:03d}" for i in range(20, 30)
    ]

    # Final state: all 30 on disk, no duplicates.
    final_rows = read_candidates_jsonl(out_path)
    assert {c.source_id for c in final_rows} == {f"src_{i:03d}" for i in range(30)}
    assert len(final_rows) == 30  # no duplicates


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3 — Multi-client parallel (4 seats, like real Path 2).
# ─────────────────────────────────────────────────────────────────────────────


def test_mock_run_four_clients_parallel(tmp_path: Path) -> None:
    """Default Path 2 lineup shape: 4 distinct seats running in parallel."""
    prompts = _make_prompts(15)
    clients = [
        _FakeModelClient(provider="anthropic", model="m/haiku"),
        _FakeModelClient(provider="openai", model="m/gpt5mini"),
        _FakeModelClient(provider="google_gemini", model="m/gemini"),
        _FakeModelClient(provider="together", model="m/maverick"),
    ]
    strategy = PaidAPIStrategy(clients=clients)
    results = strategy.run_pair("gdpr__pdpa_sg", prompts, tmp_path, smoke=False)

    assert set(results.keys()) == {"m/haiku", "m/gpt5mini", "m/gemini", "m/maverick"}
    for model in ("m/haiku", "m/gpt5mini", "m/gemini", "m/maverick"):
        rr = results[model]
        assert rr.parse_ok == 15
        assert rr.parse_errors == 0
        out_path = output_path_for(tmp_path, "gdpr__pdpa_sg", model)
        rows = read_candidates_jsonl(out_path)
        assert len(rows) == 15
        assert all(c.parse_error is None for c in rows)
        # Each row references the right model.
        assert all(c.model == model for c in rows)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 4 — Idempotent re-run after a full success.
# ─────────────────────────────────────────────────────────────────────────────


def test_mock_run_full_then_idempotent_rerun(tmp_path: Path) -> None:
    """A second run after a complete first run does zero new work."""
    prompts = _make_prompts(10)

    # First run: clean, all 10 succeed.
    client_1 = _FakeModelClient(model="t/idem")
    strategy_1 = PaidAPIStrategy(clients=[client_1])
    results_1 = strategy_1.run_pair("gdpr__pdpa_sg", prompts, tmp_path, smoke=False)
    assert results_1["t/idem"].total_processed == 10
    assert client_1.call_count == 10

    # Second run: same prompts, fresh client. Should make zero calls.
    client_2 = _FakeModelClient(model="t/idem")
    strategy_2 = PaidAPIStrategy(clients=[client_2])
    results_2 = strategy_2.run_pair("gdpr__pdpa_sg", prompts, tmp_path, smoke=False)

    assert client_2.call_count == 0  # critical: no re-calls
    rr = results_2["t/idem"]
    assert rr.total_processed == 0
    assert rr.resumed_from_disk == 10
    assert rr.parse_ok == 0  # nothing newly processed
    assert rr.parse_errors == 0

    # Output file unchanged in row count.
    out_path = output_path_for(tmp_path, "gdpr__pdpa_sg", "t/idem")
    assert len(read_candidates_jsonl(out_path)) == 10


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 5 — Crash inside a multi-client run does not poison other seats.
# ─────────────────────────────────────────────────────────────────────────────


def test_mock_run_one_seat_failure_does_not_poison_others(tmp_path: Path) -> None:
    """One client raising mid-pair → other seats still complete successfully.

    Uses an Exception subclass (not BaseException) so the failing seat's
    candidate gets recorded as parse_error rather than crashing the run.
    """
    prompts = _make_prompts(8)
    bad_responses = {f"src_{i:03d}": RuntimeError("bad seat") for i in range(8)}
    clients = [
        _FakeModelClient(model="m/good_a"),
        _FakeModelClient(model="m/bad", responses=bad_responses),
        _FakeModelClient(model="m/good_b"),
    ]
    strategy = PaidAPIStrategy(clients=clients)
    results = strategy.run_pair("gdpr__pdpa_sg", prompts, tmp_path, smoke=False)

    # Good seats finished cleanly.
    for good in ("m/good_a", "m/good_b"):
        assert results[good].parse_ok == 8
        assert results[good].parse_errors == 0
    # Bad seat recorded all 8 as parse_error.
    assert results["m/bad"].parse_ok == 0
    assert results["m/bad"].parse_errors == 8
    bad_rows = read_candidates_jsonl(output_path_for(tmp_path, "gdpr__pdpa_sg", "m/bad"))
    assert len(bad_rows) == 8
    assert all("RuntimeError" in (c.parse_error or "") for c in bad_rows)
