"""Tier-7A `EnsembleStrategy` Protocol + resilient JSONL helper tests.

Focus: the resilience contract documented in
`daccord.ensemble.strategy` — per-call durable append, resume by
`source_id`, error containment that keeps a partial run progressing.

Tests use `_FakeModelClient` (a deterministic stand-in for ModelClient)
so we can simulate any failure mode without touching real SDKs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from daccord.ensemble import BatchPrompt, EnsembleCandidate
from daccord.ensemble.strategies.paid_api import PaidAPIStrategy
from daccord.ensemble.strategy import (
    append_candidate,
    load_completed_source_ids,
    make_error_candidate,
    output_path_for,
    prune_parse_errors,
    read_candidates_jsonl,
    write_candidates_atomic,
)
from daccord.eval.schema import CitationCandidate, ModelResponse, PromptMessages


def _make_prompt(source_id: str, *, source_framework: str = "gdpr") -> BatchPrompt:
    return BatchPrompt(
        record_id=source_id,
        source_id=source_id,
        source_jurisdiction="eu",
        source_framework=source_framework,
        source_citation_id=source_id,
        source_mechanism=f"mechanism for {source_id}",
        target_jurisdiction="sg",
        target_framework="pdpa_sg",
        system="system",
        user=f"user prompt for {source_id}",
        max_tokens=256,
    )


def _make_candidate(
    source_id: str, *, model: str = "test/model", parse_error: str | None = None
) -> EnsembleCandidate:
    return EnsembleCandidate(
        source_id=source_id,
        source_jurisdiction="eu",
        source_framework="gdpr",
        source_citation_id=source_id,
        source_mechanism=f"mechanism for {source_id}",
        target_jurisdiction="sg",
        target_framework="pdpa_sg",
        model=model,
        citation_id=f"sg-{source_id}" if parse_error is None else "",
        target_mechanism=f"sg mapped {source_id}" if parse_error is None else "",
        mapping_justification="because" if parse_error is None else "",
        parse_error=parse_error,
    )


# ─────────────────────────────────────────────────────────────────────────────
# `output_path_for` + `read_candidates_jsonl` + `append_candidate` basics.
# ─────────────────────────────────────────────────────────────────────────────


def test_output_path_uses_model_slug(tmp_path: Path) -> None:
    path = output_path_for(tmp_path, "gdpr__pdpa_sg", "anthropic.claude-haiku-4-5-20251001-v1:0")
    assert path == tmp_path / "gdpr__pdpa_sg__anthropic-claude-haiku-4-5-20251001-v1-0.jsonl"


def test_read_candidates_jsonl_empty_when_missing(tmp_path: Path) -> None:
    assert read_candidates_jsonl(tmp_path / "absent.jsonl") == []


def test_append_then_read_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    candidates = [_make_candidate(f"src_{i}") for i in range(3)]
    for c in candidates:
        append_candidate(path, c)
    read_back = read_candidates_jsonl(path)
    assert len(read_back) == 3
    assert [c.source_id for c in read_back] == ["src_0", "src_1", "src_2"]


def test_load_completed_source_ids_returns_set(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    for sid in ["src_1", "src_2", "src_3"]:
        append_candidate(path, _make_candidate(sid))
    assert load_completed_source_ids(path) == {"src_1", "src_2", "src_3"}


def test_load_completed_source_ids_includes_parse_errors(tmp_path: Path) -> None:
    """parse_error rows count as 'attempted' — resume must skip them too."""
    path = tmp_path / "out.jsonl"
    append_candidate(path, _make_candidate("src_1"))
    append_candidate(path, _make_candidate("src_2", parse_error="schema failed"))
    assert load_completed_source_ids(path) == {"src_1", "src_2"}


def test_read_candidates_jsonl_skips_corrupt_rows(tmp_path: Path) -> None:
    """One corrupt row in the middle doesn't kill the read."""
    path = tmp_path / "out.jsonl"
    append_candidate(path, _make_candidate("src_1"))
    # Inject an unparseable line.
    with path.open("a", encoding="utf-8") as f:
        f.write("this is not valid json\n")
    append_candidate(path, _make_candidate("src_2"))
    rows = read_candidates_jsonl(path)
    assert {c.source_id for c in rows} == {"src_1", "src_2"}


def test_append_candidate_creates_parent_dir(tmp_path: Path) -> None:
    deep = tmp_path / "nested" / "deeper" / "out.jsonl"
    assert not deep.parent.exists()
    append_candidate(deep, _make_candidate("src_1"))
    assert deep.exists()
    assert len(read_candidates_jsonl(deep)) == 1


def test_make_error_candidate_flags_parse_error() -> None:
    prompt = _make_prompt("src_x")
    cand = make_error_candidate(prompt, model="m", error_message="timeout")
    assert cand.source_id == "src_x"
    assert cand.parse_error == "timeout"
    assert cand.citation_id == ""


def test_prune_parse_errors_removes_only_error_rows(tmp_path: Path) -> None:
    """Successful rows survive; parse_error rows are scrubbed."""
    path = tmp_path / "out.jsonl"
    append_candidate(path, _make_candidate("src_1"))
    append_candidate(path, _make_candidate("src_2", parse_error="timeout"))
    append_candidate(path, _make_candidate("src_3"))
    append_candidate(path, _make_candidate("src_4", parse_error="rate limit"))
    removed = prune_parse_errors(path)
    assert removed == 2
    remaining = read_candidates_jsonl(path)
    assert {c.source_id for c in remaining} == {"src_1", "src_3"}
    assert all(c.parse_error is None for c in remaining)


def test_prune_parse_errors_noop_when_clean(tmp_path: Path) -> None:
    """All-success file: no rewrite, returns 0."""
    path = tmp_path / "out.jsonl"
    append_candidate(path, _make_candidate("src_1"))
    append_candidate(path, _make_candidate("src_2"))
    before_mtime = path.stat().st_mtime_ns
    removed = prune_parse_errors(path)
    assert removed == 0
    # File untouched (mtime unchanged).
    assert path.stat().st_mtime_ns == before_mtime


def test_prune_parse_errors_missing_file_returns_zero(tmp_path: Path) -> None:
    assert prune_parse_errors(tmp_path / "nope.jsonl") == 0


def test_paid_api_strategy_retry_errors_recalls_only_failed_rows(tmp_path: Path) -> None:
    """`retry_errors=True` re-calls only the source_ids that had parse_errors.

    Other source_ids on the same seat — and ALL source_ids on other seats
    that succeeded — are NOT re-called. This is the "1 model fails on row X;
    other models' results stay, rerun only that 1 model on row X" semantic.
    """
    # Seat A: src_0 succeeded, src_1 had parse_error
    # Seat B: src_0 + src_1 both succeeded
    path_a = output_path_for(tmp_path, "gdpr__pdpa_sg", "m/a")
    path_b = output_path_for(tmp_path, "gdpr__pdpa_sg", "m/b")
    append_candidate(path_a, _make_candidate("src_0", model="m/a"))
    append_candidate(path_a, _make_candidate("src_1", model="m/a", parse_error="timeout"))
    append_candidate(path_b, _make_candidate("src_0", model="m/b"))
    append_candidate(path_b, _make_candidate("src_1", model="m/b"))

    client_a = _FakeModelClient(provider="pa", model="m/a")
    client_b = _FakeModelClient(provider="pb", model="m/b")
    strategy = PaidAPIStrategy(clients=[client_a, client_b])
    prompts = [_make_prompt(f"src_{i}") for i in range(2)]
    results = strategy.run_pair(
        "gdpr__pdpa_sg", prompts, tmp_path, smoke=False, retry_errors=True
    )

    # Seat A: only src_1 re-called (the parse_error one). src_0 stayed.
    assert results["m/a"].total_processed == 1
    assert results["m/a"].resumed_from_disk == 1
    assert client_a.call_count == 1
    # Seat B: nothing re-called (no parse_errors to prune).
    assert results["m/b"].total_processed == 0
    assert results["m/b"].resumed_from_disk == 2
    assert client_b.call_count == 0

    # Final state: both seats have clean 2-row files.
    rows_a = read_candidates_jsonl(path_a)
    assert len(rows_a) == 2
    assert all(c.parse_error is None for c in rows_a)
    assert {c.source_id for c in rows_a} == {"src_0", "src_1"}


def test_paid_api_strategy_default_does_not_retry_errors(tmp_path: Path) -> None:
    """Without --retry-errors, parse_error rows persist and are not re-called."""
    path = output_path_for(tmp_path, "gdpr__pdpa_sg", "m/c")
    append_candidate(path, _make_candidate("src_0", model="m/c"))
    append_candidate(path, _make_candidate("src_1", model="m/c", parse_error="timeout"))

    client = _FakeModelClient(model="m/c")
    strategy = PaidAPIStrategy(clients=[client])
    prompts = [_make_prompt(f"src_{i}") for i in range(2)]
    results = strategy.run_pair("gdpr__pdpa_sg", prompts, tmp_path, smoke=False)

    # Both source_ids were already on disk (one succeeded, one parse_error);
    # both count as completed → zero new calls.
    assert client.call_count == 0
    assert results["m/c"].resumed_from_disk == 2
    assert results["m/c"].total_processed == 0
    # The parse_error row is still on disk (unchanged).
    rows = read_candidates_jsonl(path)
    err_rows = [c for c in rows if c.parse_error is not None]
    assert len(err_rows) == 1
    assert err_rows[0].source_id == "src_1"


def test_write_candidates_atomic_sorts_by_source_id(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    candidates = [_make_candidate(s) for s in ["src_3", "src_1", "src_2"]]
    write_candidates_atomic(path, candidates)
    read_back = read_candidates_jsonl(path)
    assert [c.source_id for c in read_back] == ["src_1", "src_2", "src_3"]


# ─────────────────────────────────────────────────────────────────────────────
# `PaidAPIStrategy` — happy path + crash/resume + error containment.
# ─────────────────────────────────────────────────────────────────────────────


class _FakeModelClient:
    """Test-only ModelClient. Configurable per-prompt behaviour.

    `provider` + `model` are present so the strategy can iterate.
    `responses` is a dict source_id -> (ModelResponse | Exception). Default
    behavior for unseen source_ids: return a valid CitationCandidate.

    `crash_after` (int | None): if set, raise SystemExit on the Nth call
    (0-indexed). SystemExit is a BaseException, not Exception — so the
    strategy's `except Exception` guard does NOT catch it. That simulates
    a hard process crash (SIGKILL, Ctrl-C) rather than a recoverable
    transient error.
    """

    def __init__(
        self,
        *,
        provider: Any = "test_provider",
        model: str = "test/fake",
        responses: dict[str, Any] | None = None,
        crash_after: int | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self._responses = responses or {}
        self._crash_after = crash_after
        self._calls: list[str] = []

    @property
    def call_count(self) -> int:
        return len(self._calls)

    def generate(self, messages: PromptMessages, *, run_id: str, batch_id: str) -> ModelResponse:
        # Hard-crash simulation BEFORE recording the call, so a re-run's
        # completed-source-ids set won't contain the crash-victim.
        # Use SystemExit so the strategy's `except Exception` does NOT
        # catch it — emulating a real SIGKILL/Ctrl-C.
        if self._crash_after is not None and len(self._calls) >= self._crash_after:
            raise SystemExit("CRASH")

        # Extract source_id from the user prompt (test convention: prompts
        # built via _make_prompt put it in the user string).
        source_id = self._extract_source_id(messages.user)
        self._calls.append(source_id)

        configured = self._responses.get(source_id)
        if configured is not None:
            if isinstance(configured, Exception):
                raise configured
            return configured

        # Default: happy path.
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

    @staticmethod
    def _extract_source_id(user_prompt: str) -> str:
        # Test prompts are "user prompt for src_X" — last token is source_id.
        return user_prompt.split()[-1]


def test_paid_api_strategy_happy_path(tmp_path: Path) -> None:
    client = _FakeModelClient(model="t/happy")
    strategy = PaidAPIStrategy(clients=[client])
    prompts = [_make_prompt(f"src_{i}") for i in range(5)]
    results = strategy.run_pair("gdpr__pdpa_sg", prompts, tmp_path, smoke=False)

    assert set(results.keys()) == {"t/happy"}
    rr = results["t/happy"]
    assert rr.parse_ok == 5
    assert rr.parse_errors == 0
    assert rr.resumed_from_disk == 0
    assert rr.total_processed == 5

    # All 5 candidates landed on disk.
    out_path = output_path_for(tmp_path, "gdpr__pdpa_sg", "t/happy")
    rows = read_candidates_jsonl(out_path)
    assert {c.source_id for c in rows} == {f"src_{i}" for i in range(5)}
    assert all(c.parse_error is None for c in rows)


def test_paid_api_strategy_records_parse_error_for_response_errors(tmp_path: Path) -> None:
    """A ModelResponse with parse_error is recorded as an attempted-failed row."""
    bad_response = ModelResponse(
        model="t/bad",
        top1=None,
        raw_text="not json",
        input_tokens=10,
        output_tokens=0,
        latency_ms=5.0,
        parse_error="json decode failed",
    )
    client = _FakeModelClient(model="t/bad", responses={"src_2": bad_response})
    strategy = PaidAPIStrategy(clients=[client])
    prompts = [_make_prompt(f"src_{i}") for i in range(4)]
    results = strategy.run_pair("gdpr__pdpa_sg", prompts, tmp_path, smoke=False)

    rr = results["t/bad"]
    assert rr.parse_ok == 3
    assert rr.parse_errors == 1

    out_path = output_path_for(tmp_path, "gdpr__pdpa_sg", "t/bad")
    rows = read_candidates_jsonl(out_path)
    err_rows = [c for c in rows if c.parse_error is not None]
    assert len(err_rows) == 1
    assert err_rows[0].source_id == "src_2"
    assert "json decode" in (err_rows[0].parse_error or "")


def test_paid_api_strategy_records_parse_error_for_exceptions(tmp_path: Path) -> None:
    """An exception during generate() is caught and recorded, not propagated."""
    client = _FakeModelClient(
        model="t/exc",
        responses={"src_1": TimeoutError("network died")},
    )
    strategy = PaidAPIStrategy(clients=[client])
    prompts = [_make_prompt(f"src_{i}") for i in range(3)]
    results = strategy.run_pair("gdpr__pdpa_sg", prompts, tmp_path, smoke=False)

    rr = results["t/exc"]
    assert rr.parse_ok == 2  # src_0 and src_2 succeeded
    assert rr.parse_errors == 1

    out_path = output_path_for(tmp_path, "gdpr__pdpa_sg", "t/exc")
    rows = read_candidates_jsonl(out_path)
    err = [c for c in rows if c.source_id == "src_1"][0]
    assert "TimeoutError" in (err.parse_error or "")
    assert "network died" in (err.parse_error or "")


def test_paid_api_strategy_resumes_skipping_completed(tmp_path: Path) -> None:
    """Pre-existing rows on disk are not re-called."""
    out_path = output_path_for(tmp_path, "gdpr__pdpa_sg", "t/resume")
    # Pre-populate src_0 and src_1.
    for sid in ["src_0", "src_1"]:
        append_candidate(out_path, _make_candidate(sid, model="t/resume"))

    client = _FakeModelClient(model="t/resume")
    strategy = PaidAPIStrategy(clients=[client])
    prompts = [_make_prompt(f"src_{i}") for i in range(5)]
    results = strategy.run_pair("gdpr__pdpa_sg", prompts, tmp_path, smoke=False)

    rr = results["t/resume"]
    assert rr.resumed_from_disk == 2
    assert rr.total_processed == 3  # src_2, src_3, src_4
    assert rr.parse_ok == 3
    assert client.call_count == 3  # critical: src_0, src_1 NOT re-called

    rows = read_candidates_jsonl(out_path)
    assert {c.source_id for c in rows} == {f"src_{i}" for i in range(5)}


def test_paid_api_strategy_survives_mid_run_crash(tmp_path: Path) -> None:
    """Crash mid-run: completed rows persist, next invocation resumes cleanly."""
    out_path = output_path_for(tmp_path, "gdpr__pdpa_sg", "t/crash")
    prompts = [_make_prompt(f"src_{i}") for i in range(5)]

    # Run 1: crash after 2 successful calls.
    crashing_client = _FakeModelClient(model="t/crash", crash_after=2)
    strategy_1 = PaidAPIStrategy(clients=[crashing_client])
    with pytest.raises(SystemExit, match="CRASH"):
        # Call the per-client method directly so the BaseException
        # propagates through the test (bypassing the ThreadPoolExecutor
        # which would otherwise mask it).
        strategy_1._run_one_client(  # noqa: SLF001
            client=crashing_client,
            framework_pair="gdpr__pdpa_sg",
            prompts=prompts,
            out_dir=tmp_path,
        )

    # 2 rows should be on disk after the crash.
    rows_after_crash = read_candidates_jsonl(out_path)
    assert len(rows_after_crash) == 2
    completed_ids = {c.source_id for c in rows_after_crash}
    assert completed_ids == {"src_0", "src_1"}

    # Run 2: fresh client, no crash. Should pick up where run 1 left off.
    fresh_client = _FakeModelClient(model="t/crash")
    strategy_2 = PaidAPIStrategy(clients=[fresh_client])
    results = strategy_2.run_pair("gdpr__pdpa_sg", prompts, tmp_path, smoke=False)

    rr = results["t/crash"]
    assert rr.resumed_from_disk == 2
    assert rr.total_processed == 3
    assert rr.parse_ok == 3
    assert fresh_client.call_count == 3  # only src_2, src_3, src_4

    final_rows = read_candidates_jsonl(out_path)
    assert {c.source_id for c in final_rows} == {f"src_{i}" for i in range(5)}


def test_paid_api_strategy_rejects_duplicate_models() -> None:
    """Two clients with the same model id would race the output file."""
    c1 = _FakeModelClient(model="same/model")
    c2 = _FakeModelClient(model="same/model")
    with pytest.raises(ValueError, match="duplicate model id"):
        PaidAPIStrategy(clients=[c1, c2])


def test_paid_api_strategy_rejects_empty_clients() -> None:
    with pytest.raises(ValueError, match="at least one"):
        PaidAPIStrategy(clients=[])


def test_paid_api_strategy_multi_client_runs_in_parallel(tmp_path: Path) -> None:
    """Two clients = two output files, both populated."""
    c1 = _FakeModelClient(provider="p1", model="m/one")
    c2 = _FakeModelClient(provider="p2", model="m/two")
    strategy = PaidAPIStrategy(clients=[c1, c2])
    prompts = [_make_prompt(f"src_{i}") for i in range(3)]
    results = strategy.run_pair("gdpr__pdpa_sg", prompts, tmp_path, smoke=False)

    assert set(results.keys()) == {"m/one", "m/two"}
    for model in ("m/one", "m/two"):
        out_path = output_path_for(tmp_path, "gdpr__pdpa_sg", model)
        rows = read_candidates_jsonl(out_path)
        assert len(rows) == 3
        assert all(c.parse_error is None for c in rows)


def test_paid_api_strategy_models_property_matches_clients() -> None:
    c1 = _FakeModelClient(model="alpha")
    c2 = _FakeModelClient(model="beta", provider="other")
    strategy = PaidAPIStrategy(clients=[c1, c2])
    assert strategy.models == ["alpha", "beta"]
    assert strategy.name == "local-api-paid"
