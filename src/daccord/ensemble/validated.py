"""Tier-7C hand-validation overlay (consumed by tier-7B splits + tier-9 gold).

The tier-7C labeler (`consumer/labeler/app.py`) writes one `ValidatedPair`
row per reviewed source clause to
`data/ensemble/validated/{framework_pair}.jsonl`. The splits builder
([src/daccord/ensemble/splits.py](splits.py)) merges these overlays on
top of tier-6B output to decide gold eligibility.

Same write-once-per-source_id invariant as the raw layer
([src/daccord/ensemble/strategy.py](strategy.py)): a row, once reviewed,
is locked. Re-reviewing requires deleting the file (or pruning that
source_id manually) — guards against accidental clobber across
reviewers / sessions.

This module lives separately from `strategy.py` because the raw-layer
immutability rule is its own contract; the overlay layer is a different
contract for a different file tree. Both raise the shared
`ImmutabilityViolation` exception when violated.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from daccord.ensemble.schema import Tier
from daccord.ensemble.strategy import ImmutabilityViolation
from daccord.validation import ValidatedModel, validated


class ValidatedPair(ValidatedModel):
    """One reviewer's verdict on one tiered pair.

    `chosen_citation_id`:
      - non-empty → the target citation the reviewer endorses (may match
        the model consensus, a dissenting vote, or a free-form ID).
      - empty string → reviewer confirmed "no analog exists" for this
        source clause in the target framework. Excludes the row from
        gold even if the model assigned HIGH.

    `reviewer` + `reviewed_at` are audit fields; not used by splits but
    preserved for tier-9 manifest reproducibility.
    """

    source_id: str
    source_jurisdiction: str
    source_framework: str
    target_jurisdiction: str
    target_framework: str
    chosen_citation_id: str
    human_note: str = ""
    reviewer: str
    reviewed_at: str  # ISO-8601 UTC
    tier_at_review: Tier
    agreement_score_at_review: float


@validated
def append_validation(path: Path, row: ValidatedPair) -> None:
    """Append one `ValidatedPair` row to `path` durably + immutably.

    Refuses if `path` already contains a row with the same `source_id` —
    a reviewed pair is locked. Per-row `flush()` + `os.fsync()` so the
    row is on disk before the call returns.
    """
    if path.exists():
        existing = load_validated_source_ids(path)
        if row.source_id in existing:
            raise ImmutabilityViolation(
                f"refusing to append duplicate source_id={row.source_id!r} to "
                f"{path.name}: a reviewed pair is locked. Delete the file or "
                f"manually prune that row if a re-review is intentional."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    line = row.model_dump_json() + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


@validated
def load_validated_source_ids(path: Path) -> set[str]:
    """Return the set of `source_id`s already reviewed in `path`.

    Used by the labeler at load time to filter the queue down to
    unreviewed pairs (resume-by-source_id).
    """
    if not path.exists():
        return set()
    ids: set[str] = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        sid = obj.get("source_id")
        if not isinstance(sid, str):
            raise ValueError(f"{path}:{lineno}: missing/invalid source_id")
        ids.add(sid)
    return ids


@validated
def read_validations(path: Path) -> list[ValidatedPair]:
    """Read all `ValidatedPair` rows from `path` (empty list if absent)."""
    if not path.exists():
        return []
    rows: list[ValidatedPair] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            rows.append(ValidatedPair.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"{path}:{lineno}: invalid ValidatedPair row: {exc}") from exc
    return rows
