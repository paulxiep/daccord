"""Tier-7C labeler queue logic (separated from Streamlit UI for testing).

Loads tier-6B tiered output, merges the tier-6B+ bidirectional cross-check
overlay (when present), applies the validated overlay to filter out
already-reviewed source_ids, sorts the remainder, and returns a queue of
`QueuedPair` rows ready for the next review action.

`app.py` wraps this with Streamlit widgets; this module is pure data
plumbing and has no Streamlit dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from daccord.ensemble.bidirectional import (
    BidirectionalResult,
    BidirectionalStatus,
    load_bidirectional_for_pair,
)
from daccord.ensemble.schema import Tier, TieredPair
from daccord.ensemble.validated import load_validated_source_ids
from daccord.validation import ValidatedModel, validated

# Sort orders the operator can pick in the sidebar.
SortOrder = Literal["confidence-desc", "confidence-asc", "source-id"]


@dataclass(frozen=True)
class QueuedPair:
    """A tiered pair plus its bidirectional status (for UI display + filter)."""

    pair: TieredPair
    bidirectional: BidirectionalResult | None


class PairFile(ValidatedModel):
    """One framework-pair on disk + its review state."""

    name: str
    tiered_path: Path
    validated_path: Path
    bidirectional_path: Path
    total: int
    reviewed: int
    bidirectional_consistent: int
    bidirectional_inconsistent: int


@validated
def list_pair_files(
    tiered_dir: Path, validated_dir: Path, bidirectional_dir: Path | None = None
) -> list[PairFile]:
    """Enumerate framework-pairs available for review + their progress."""
    files: list[PairFile] = []
    for tiered_path in sorted(tiered_dir.glob("*.jsonl")):
        name = tiered_path.stem
        validated_path = validated_dir / f"{name}.jsonl"
        bidirectional_path = (
            (bidirectional_dir / f"{name}.jsonl")
            if bidirectional_dir is not None
            else Path("/dev/null/never-exists")
        )
        total = _count_jsonl_lines(tiered_path)
        reviewed = len(load_validated_source_ids(validated_path))
        bi_overlay = load_bidirectional_for_pair(bidirectional_dir, name)
        bi_consistent = sum(1 for r in bi_overlay.values() if r.status == "consistent")
        bi_inconsistent = sum(1 for r in bi_overlay.values() if r.status == "inconsistent")
        files.append(
            PairFile(
                name=name,
                tiered_path=tiered_path,
                validated_path=validated_path,
                bidirectional_path=bidirectional_path,
                total=total,
                reviewed=reviewed,
                bidirectional_consistent=bi_consistent,
                bidirectional_inconsistent=bi_inconsistent,
            )
        )
    return files


@validated
def build_review_queue(
    tiered_path: Path,
    validated_path: Path,
    *,
    bidirectional_dir: Path | None = None,
    tier_filter: list[Tier],
    bidirectional_filter: list[BidirectionalStatus] | None = None,
    sort_order: SortOrder = "confidence-desc",
) -> list[QueuedPair]:
    """Return the unreviewed queue for a pair, filtered + sorted.

    `tier_filter`: which forward tiers to surface (HIGH usually goes
    through stratified spot-check separately; default = MED + LOW + SALVAGE).

    `bidirectional_filter`: which bidirectional statuses to surface. Common
    settings:
      - None: don't filter on bidirectional (show all).
      - ["inconsistent", "reverse_unknown", "missing_in_registry",
        "missing_reverse_row", "missing_reverse_pair",
        "no_forward_consensus"]: skip auto-promotable `consistent` rows;
        focus reviewer time on the cases where bidirectional doesn't help.

    `sort_order`:
      - `confidence-desc` — highest `agreement_score` first. Surfaces the
        easiest MED rows first (high-throughput); SALVAGE (-1.0) lands last.
      - `confidence-asc` — lowest score first. Useful when intentionally
        reviewing edge cases.
      - `source-id` — deterministic ordering (mostly for tests).
    """
    pair_name = tiered_path.stem
    bi_overlay = load_bidirectional_for_pair(bidirectional_dir, pair_name)
    reviewed = load_validated_source_ids(validated_path)
    all_rows = _read_tiered(tiered_path)
    tier_set = set(tier_filter)
    bi_filter_set = set(bidirectional_filter) if bidirectional_filter is not None else None

    queue: list[QueuedPair] = []
    for r in all_rows:
        if r.tier not in tier_set:
            continue
        if r.source_id in reviewed:
            continue
        bi = bi_overlay.get(r.source_id)
        if bi_filter_set is not None:
            bi_status: BidirectionalStatus = bi.status if bi else "no_forward_consensus"
            if bi_status not in bi_filter_set:
                continue
        queue.append(QueuedPair(pair=r, bidirectional=bi))

    return _sort_queue(queue, sort_order)


def _sort_queue(queue: list[QueuedPair], order: SortOrder) -> list[QueuedPair]:
    if order == "confidence-desc":
        return sorted(queue, key=lambda q: (-q.pair.agreement_score, q.pair.source_id))
    if order == "confidence-asc":
        return sorted(queue, key=lambda q: (q.pair.agreement_score, q.pair.source_id))
    return sorted(queue, key=lambda q: q.pair.source_id)


def _read_tiered(path: Path) -> list[TieredPair]:
    if not path.exists():
        return []
    rows: list[TieredPair] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            rows.append(TieredPair.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"{path}:{lineno}: invalid TieredPair row: {exc}") from exc
    return rows


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for raw in path.read_text(encoding="utf-8").splitlines() if raw.strip())


@validated
def vote_choices(row: TieredPair) -> list[tuple[str, str]]:
    """Return `(value, label)` pairs for the reviewer's radio.

    Includes one entry per distinct non-empty vote (deduplicated by
    normalized citation_id, showing the raw form + model count), plus
    a sentinel for "no analog" and a free-form "other" prompt.
    """
    seen: dict[str, list[str]] = {}
    raw_for_norm: dict[str, str] = {}
    for v in row.votes:
        nid = v.citation_id_normalized
        if not nid:
            continue
        seen.setdefault(nid, []).append(v.model)
        raw_for_norm.setdefault(nid, v.citation_id_raw)
    choices: list[tuple[str, str]] = []
    for nid, models in sorted(seen.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        plural = "s" if len(models) > 1 else ""
        label = f"{raw_for_norm[nid]}  ({len(models)} vote{plural}: {', '.join(models)})"
        choices.append((nid, label))
    choices.append(("", "No analog exists in target framework"))
    return choices


@validated
def progress_summary(files: list[PairFile]) -> dict[str, int]:
    """Aggregate counts across all pair files for the sidebar header."""
    total = sum(f.total for f in files)
    reviewed = sum(f.reviewed for f in files)
    return {"total": total, "reviewed": reviewed, "remaining": total - reviewed}
