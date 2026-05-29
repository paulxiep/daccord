"""Tier-6B+ bidirectional cross-check.

The 72 framework-pairs run in tier 7A cover both directions of every framework
pair (e.g. both `gdpr__pdpa_sg` and `pdpa_sg__gdpr`). For each forward tiered
row, we can look up whether the reverse direction agrees:

  - Forward `A.cs → B.ct` matches reverse `B.ct → A.cs` → **consistent**
    (strong evidence the mapping is real; can auto-promote MED → gold).
  - Forward `A.cs → B.ct` but reverse `B.ct → A.cs'` with `A.cs ≠ A.cs'` →
    **inconsistent** (worth hand-val even at HIGH confidence).
  - Reverse direction is LOW/SALVAGE → **reverse_unknown** (no signal).
  - Forward consensus didn't resolve to any reverse source clause →
    **missing_in_registry** (consensus citation_id hallucinated or
    normalised to something the reverse framework doesn't index).
  - Reverse pair file or row missing → **missing_reverse_pair /
    _row**.
  - Forward was SALVAGE → **no_forward_consensus** (nothing to check).

The framework's clause→source_id mapping is implicit in the raw files:
each row records `(source_framework, source_citation_id, source_id)`, so
`build_framework_lookup` reads one raw file per framework and builds
`{normalize(source_citation_id): source_id}`. This avoids coupling to
[data/clauses/](data/clauses/) shape.

Output lives in `data/ensemble/bidirectional/{forward_pair}.jsonl`,
keyed by forward `source_id`. Tier 7B splits + the tier-7C labeler both
merge this overlay on top of tier-6B output.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Literal

from daccord.ensemble.schema import EnsembleCandidate, Tier, TieredPair
from daccord.eval.scoring import normalize_citation_id
from daccord.validation import ValidatedModel, validated

BidirectionalStatus = Literal[
    "consistent",
    "inconsistent",
    "reverse_unknown",
    "missing_reverse_row",
    "missing_reverse_pair",
    "missing_in_registry",
    "no_forward_consensus",
]


class BidirectionalResult(ValidatedModel):
    """Cross-direction agreement for one forward tiered row."""

    source_id: str
    forward_consensus: str
    reverse_pair: str
    reverse_source_id: str | None
    reverse_consensus: str | None
    reverse_tier: Tier | None
    reverse_agreement_score: float | None
    status: BidirectionalStatus


@validated
def reverse_pair_name(forward_pair: str) -> str:
    """`gdpr__pdpa_sg` → `pdpa_sg__gdpr`."""
    src, tgt = forward_pair.split("__", 1)
    return f"{tgt}__{src}"


@validated
def build_framework_lookup(framework: str, raw_dir: Path) -> dict[str, str]:
    """Read one raw file for `framework` and return `{normalize(source_citation_id): source_id}`.

    Any `{framework}__*.jsonl` works — every row for a framework-as-source
    repeats the same `(source_id, source_citation_id)` pairs (one row per
    target × seat combination). Reading a single file is enough.
    """
    files = sorted(raw_dir.glob(f"{framework}__*.jsonl"))
    if not files:
        return {}
    lookup: dict[str, str] = {}
    for raw_line in files[0].read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        cand = EnsembleCandidate.model_validate_json(line)
        if not cand.source_citation_id:
            continue
        norm = normalize_citation_id(cand.source_citation_id)
        if norm and norm not in lookup:
            lookup[norm] = cand.source_id
    return lookup


@validated
def compute_bidirectional_for_pair(
    forward_pair: str,
    tiered_dir: Path,
    raw_dir: Path,
) -> list[BidirectionalResult]:
    """Compute bidirectional cross-check for all rows in `forward_pair`."""
    forward_path = tiered_dir / f"{forward_pair}.jsonl"
    if not forward_path.exists():
        raise FileNotFoundError(f"forward tiered file missing: {forward_path}")

    forward_rows = _read_tiered(forward_path)
    reverse_pair = reverse_pair_name(forward_pair)

    if not forward_rows:
        return []

    target_framework = forward_rows[0].target_framework
    reverse_path = tiered_dir / f"{reverse_pair}.jsonl"

    if not reverse_path.exists():
        return [
            _result(
                row,
                reverse_pair=reverse_pair,
                reverse_source_id=None,
                reverse_consensus=None,
                reverse_tier=None,
                reverse_agreement_score=None,
                status="missing_reverse_pair",
            )
            for row in forward_rows
        ]

    reverse_rows_by_id: dict[str, TieredPair] = {r.source_id: r for r in _read_tiered(reverse_path)}
    target_lookup = build_framework_lookup(target_framework, raw_dir)

    return [_classify(fwd, reverse_pair, target_lookup, reverse_rows_by_id) for fwd in forward_rows]


def _classify(
    fwd: TieredPair,
    reverse_pair: str,
    target_lookup: dict[str, str],
    reverse_rows_by_id: dict[str, TieredPair],
) -> BidirectionalResult:
    if not fwd.consensus_citation_id:
        return _result(
            fwd,
            reverse_pair=reverse_pair,
            reverse_source_id=None,
            reverse_consensus=None,
            reverse_tier=None,
            reverse_agreement_score=None,
            status="no_forward_consensus",
        )

    reverse_source_id = target_lookup.get(fwd.consensus_citation_id)
    if reverse_source_id is None:
        return _result(
            fwd,
            reverse_pair=reverse_pair,
            reverse_source_id=None,
            reverse_consensus=None,
            reverse_tier=None,
            reverse_agreement_score=None,
            status="missing_in_registry",
        )

    reverse_row = reverse_rows_by_id.get(reverse_source_id)
    if reverse_row is None:
        return _result(
            fwd,
            reverse_pair=reverse_pair,
            reverse_source_id=reverse_source_id,
            reverse_consensus=None,
            reverse_tier=None,
            reverse_agreement_score=None,
            status="missing_reverse_row",
        )

    if not reverse_row.consensus_citation_id or reverse_row.tier in ("LOW", "SALVAGE"):
        # Reverse direction couldn't reach a confident consensus. Surface the
        # raw signal anyway (the labeler may display "reverse said X (LOW)"
        # as a soft hint) but don't promote.
        return _result(
            fwd,
            reverse_pair=reverse_pair,
            reverse_source_id=reverse_source_id,
            reverse_consensus=reverse_row.consensus_citation_id or None,
            reverse_tier=reverse_row.tier,
            reverse_agreement_score=reverse_row.agreement_score,
            status="reverse_unknown",
        )

    expected_back = normalize_citation_id(fwd.source_citation_id) if fwd.source_citation_id else ""
    status: BidirectionalStatus = (
        "consistent" if reverse_row.consensus_citation_id == expected_back else "inconsistent"
    )
    return _result(
        fwd,
        reverse_pair=reverse_pair,
        reverse_source_id=reverse_source_id,
        reverse_consensus=reverse_row.consensus_citation_id,
        reverse_tier=reverse_row.tier,
        reverse_agreement_score=reverse_row.agreement_score,
        status=status,
    )


def _result(
    fwd: TieredPair,
    *,
    reverse_pair: str,
    reverse_source_id: str | None,
    reverse_consensus: str | None,
    reverse_tier: Tier | None,
    reverse_agreement_score: float | None,
    status: BidirectionalStatus,
) -> BidirectionalResult:
    return BidirectionalResult(
        source_id=fwd.source_id,
        forward_consensus=fwd.consensus_citation_id,
        reverse_pair=reverse_pair,
        reverse_source_id=reverse_source_id,
        reverse_consensus=reverse_consensus,
        reverse_tier=reverse_tier,
        reverse_agreement_score=reverse_agreement_score,
        status=status,
    )


def _read_tiered(path: Path) -> list[TieredPair]:
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


@validated
def load_bidirectional_overlay(
    bidirectional_dir: Path | None,
) -> dict[str, dict[str, BidirectionalResult]]:
    """Load all bidirectional files into a nested dict keyed by forward_pair → source_id.

    Returns empty dict if the directory doesn't exist (callers treat that
    as "no bidirectional signal available").
    """
    if bidirectional_dir is None or not bidirectional_dir.exists():
        return {}
    overlay: dict[str, dict[str, BidirectionalResult]] = defaultdict(dict)
    for path in sorted(bidirectional_dir.glob("*.jsonl")):
        forward_pair = path.stem
        for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = BidirectionalResult.model_validate_json(line)
            except Exception as exc:
                raise ValueError(
                    f"{path}:{lineno}: invalid BidirectionalResult row: {exc}"
                ) from exc
            overlay[forward_pair][row.source_id] = row
    return dict(overlay)


@validated
def load_bidirectional_for_pair(
    bidirectional_dir: Path | None, forward_pair: str
) -> dict[str, BidirectionalResult]:
    """Load one pair's bidirectional file → `{source_id: BidirectionalResult}`."""
    if bidirectional_dir is None:
        return {}
    path = bidirectional_dir / f"{forward_pair}.jsonl"
    if not path.exists():
        return {}
    overlay: dict[str, BidirectionalResult] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            row = BidirectionalResult.model_validate_json(line)
        except Exception as exc:
            raise ValueError(f"{path}:{lineno}: invalid BidirectionalResult row: {exc}") from exc
        overlay[row.source_id] = row
    return overlay
