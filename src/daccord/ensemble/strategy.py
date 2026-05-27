"""Tier-7A `EnsembleStrategy` Protocol + shared resilient JSONL helpers.

Each strategy backend (Path 1 Bedrock in `strategies/bedrock.py`, Path 2 paid
direct API in `strategies/paid_api.py`, future Path 3 free-tier ensemble)
implements `EnsembleStrategy.run_pair()` and writes
`data/ensemble/raw/{framework_pair}__{model_slug}.jsonl` rows that tier 6B
consumes unchanged.

## Resilience contract — shared by all strategies

Strategies that emit candidates per call (vs Bedrock batch which lands one
file per pair) use these helpers to make every run resumable:

  - `output_path_for(out_dir, framework_pair, model)` — canonical landing.
  - `load_completed_source_ids(path)` — fast set of `source_id`s already on
    disk (so a re-run skips them).
  - `append_candidate(path, candidate)` — per-call O_APPEND write with a
    `flush()` + `fsync()` so a crash mid-call doesn't lose preceding rows.
  - `read_candidates_jsonl(path)` — load existing rows (used at start of
    run + at end for stats).

A strategy's `run_pair` should:

  1. Compute `output_path = output_path_for(out_dir, pair, model)`.
  2. `completed = load_completed_source_ids(output_path)`.
  3. Filter `prompts` to `[p for p in prompts if p.source_id not in completed]`.
  4. For each remaining prompt:
       try: candidate = call_model(prompt) ; append_candidate(...)
       except (timeout, transient): retry with backoff (per-client handles it)
       except Exception as exc: append a parse_error candidate (keeps the
         row count stable so re-runs don't pick the failed source up again
         unless explicitly retried).
  5. Return `RunResult` with counts.

The append-per-call + skip-by-source_id pattern means crashes lose at most
one in-flight call. SIGTERM and user Ctrl-C are both safe.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Protocol, runtime_checkable

from daccord.ensemble.prompt import BatchPrompt, model_slug
from daccord.ensemble.schema import EnsembleCandidate
from daccord.validation import ValidatedModel, validated

log = logging.getLogger("daccord.ensemble.strategy")


# ─────────────────────────────────────────────────────────────────────────────
# Output-path + resilient JSONL helpers (used by Path 2 + future Path 3).
# ─────────────────────────────────────────────────────────────────────────────


@validated
def output_path_for(out_dir: Path, framework_pair: str, model: str) -> Path:
    """`data/ensemble/raw/{pair}__{model_slug}.jsonl` — canonical landing path.

    Same convention as the existing Bedrock-batch path. Strategies must use
    this helper so tier 6B reads from a single well-known location.
    """
    slug = model_slug(model)
    return out_dir / f"{framework_pair}__{slug}.jsonl"


@validated
def read_candidates_jsonl(path: Path) -> list[EnsembleCandidate]:
    """Load all `EnsembleCandidate` rows from `path`. Returns [] if missing."""
    if not path.exists():
        return []
    out: list[EnsembleCandidate] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            out.append(EnsembleCandidate.model_validate_json(stripped))
        except Exception as exc:
            # Don't crash a long run on one corrupt row; log + skip. The
            # source_id won't be in `load_completed_source_ids`, so the
            # next pass will re-try it.
            log.warning("[strategy] skipping unparseable row in %s: %s", path, exc)
    return out


@validated
def load_completed_source_ids(path: Path) -> set[str]:
    """Set of `source_id`s already persisted to `path`. Empty when missing.

    Used by resumable strategies to skip work already done. Per-row parse
    failures (`parse_error != None`) DO count as completed — they represent
    a deterministic outcome of a model call. To force retry, the operator
    deletes the row from the JSONL (or the whole file).
    """
    return {c.source_id for c in read_candidates_jsonl(path)}


@validated
def append_candidate(path: Path, candidate: EnsembleCandidate) -> None:
    """Append one `EnsembleCandidate` row to `path` durably.

    Opens the file in append mode, writes the JSON line, then `flush()` +
    `os.fsync()` so the row is on the disk's storage layer before we return.
    Crashes after this point cannot lose the row.

    Path's parent is created on first call.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = candidate.model_dump_json() + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


@validated
def make_error_candidate(prompt: BatchPrompt, model: str, error_message: str) -> EnsembleCandidate:
    """Build a `parse_error`-flagged `EnsembleCandidate` for a failed call.

    Used by strategies after non-retriable exceptions so the source_id is
    recorded as "attempted, failed" — the next resumed run will skip it,
    keeping ensemble agreement counts stable.
    """
    return EnsembleCandidate(
        source_id=prompt.source_id,
        source_jurisdiction=prompt.source_jurisdiction,
        source_framework=prompt.source_framework,
        source_citation_id=prompt.source_citation_id,
        source_mechanism=prompt.source_mechanism,
        target_jurisdiction=prompt.target_jurisdiction,
        target_framework=prompt.target_framework,
        model=model,
        citation_id="",
        target_mechanism="",
        mapping_justification="",
        parse_error=error_message,
    )


@validated
def write_candidates_atomic(path: Path, candidates: list[EnsembleCandidate]) -> None:
    """Atomic full-file write (temp + replace) sorted by `source_id`.

    Used by Path 1 Bedrock-batch where all candidates land together at
    download-and-parse time, and by tests that need byte-identical rewrites.
    Per-call append (`append_candidate`) is the resilient path; this one
    is for "I have the whole list and want it sorted on disk."
    """
    sorted_rows = sorted(candidates, key=lambda c: c.source_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
        suffix=path.suffix + ".tmp",
    ) as tmp:
        for row in sorted_rows:
            tmp.write(row.model_dump_json())
            tmp.write("\n")
        tmp_name = tmp.name
    Path(tmp_name).replace(path)


@validated
def prune_parse_errors(path: Path) -> int:
    """Rewrite `path` keeping only `parse_error is None` rows. Returns the
    number of parse_error rows removed.

    Used by the `--retry-errors` resume path: scrubbing parse_error rows
    before computing `load_completed_source_ids` means the next resume
    re-calls every previously-failed source_id. Successful rows stay
    intact so a retry never duplicates work.

    Idempotent: a path with zero parse_error rows is a no-op (no file
    rewrite, no fsync); returns 0.
    """
    if not path.exists():
        return 0
    rows = read_candidates_jsonl(path)
    keep = [r for r in rows if r.parse_error is None]
    removed = len(rows) - len(keep)
    if removed == 0:
        return 0
    write_candidates_atomic(path, keep)
    log.info(
        "[strategy] pruned %d parse_error row(s) from %s; %d kept",
        removed,
        path.name,
        len(keep),
    )
    return removed


# ─────────────────────────────────────────────────────────────────────────────
# `RunResult` + `EnsembleStrategy` Protocol.
# ─────────────────────────────────────────────────────────────────────────────


class RunResult(ValidatedModel):
    """Per-(pair, model) run outcome.

    `parse_ok` + `parse_errors` add up to `total_processed` for the seat;
    they exclude rows that were already on disk from a prior partial run
    (those count in `resumed_from_disk`).
    """

    framework_pair: str
    model: str
    output_path: str
    total_processed: int  # calls actually made this run
    parse_ok: int
    parse_errors: int
    resumed_from_disk: int  # rows already present at run start (not re-called)
    seconds_elapsed: float


@runtime_checkable
class EnsembleStrategy(Protocol):
    """Single entry-point a tier-7A backend must implement.

    `name` is the CLI strategy ID (`"bedrock-batch"`, `"bedrock-sync"`,
    `"local-api-paid"`, ...) — also tagged onto MLflow + log lines.

    `models` is the ordered list of model identifiers this strategy will
    fan a pair across. For Bedrock these are the F9 model IDs from
    `m2.F9_BEDROCK_MODELS`; for paid direct API these are the
    `claude-haiku-4-5` / `gpt-5-mini` / `gemini-3.1-flash-lite` /
    `meta-llama/Llama-4-Maverick-...` strings.

    `run_pair(pair, prompts, out_dir, *, smoke)` writes one JSONL per
    model under `out_dir` and returns a `dict[model_id, RunResult]`.
    Implementations MUST use the resilience helpers above so a partial
    run can be resumed by re-invoking the same call.
    """

    name: str
    models: list[str]

    def run_pair(
        self,
        framework_pair: str,
        prompts: list[BatchPrompt],
        out_dir: Path,
        *,
        smoke: bool,
    ) -> dict[str, RunResult]: ...
