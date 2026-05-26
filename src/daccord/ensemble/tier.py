"""Tier-6B HIGH/MED/LOW/SALVAGE classifier.

Reads ensemble candidates grouped by source clause, scores 4-way agreement on
the normalized citation_id, and emits one `TieredPair` per source clause.

Rules (per [docs/development_plan.md](../../../docs/development_plan.md) tiers 6B/8):

  - **HIGH** = all N models agree on normalized citation_id (unanimous and
    non-empty). Promoted directly to gold candidate pool.
  - **MED**  = strict majority (≥ ⌈N/2⌉ + 1 for even N; ≥ ⌈N/2⌉ for odd N)
    agree on normalized citation_id, OR all N agree on the parent article
    (sub-clause split — e.g. all return Art 32 family, but split among
    32, 32(1), 32(1)(a)). Hand-validated 100% at tier 8.
  - **LOW**  = no majority and no parent agreement. Hand-validated 100% at
    tier 8; usually downgraded to eval seed rather than train.
  - **SALVAGE** = at least N-1 models returned empty citation_id (model
    explicitly said "no analog exists") OR parse errors dominate. Tier 8
    reviewer decides: either confirm the no-analog finding, or repair if
    one valid model output exists.

Parent extraction: a registry ID like "32(1)(a)" has parent "32". "5A" has
parent "5A" (no sub-clause). "มาตรา 19" (Thai) has parent "มาตรา 19" — we
only split on "(", not language-aware tokenization. This is intentional
since the sub-clause-split case is mostly a Latin-script (EU/UK) phenomenon.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from daccord.ensemble.schema import EnsembleCandidate, ModelVote, Tier, TieredPair
from daccord.eval.scoring import normalize_citation_id
from daccord.validation import validated

# Parent extraction: take the leading portion before the first "(".
# "32(1)(a)" → "32"; "5A" → "5A"; "มาตรา 19" → "มาตรา 19".
_PARENT_SPLIT = re.compile(r"^([^(]+?)\s*(?:\(.*)?$")


def _extract_parent(normalized_id: str) -> str:
    if not normalized_id:
        return ""
    m = _PARENT_SPLIT.match(normalized_id)
    return m.group(1).strip() if m else normalized_id


@validated
def classify_tier(candidates: list[EnsembleCandidate]) -> TieredPair:
    """Score one source clause's N model candidates → TieredPair.

    All candidates MUST share the same source_id, source_*, and target_*
    fields. The caller (tier_framework_pair) groups by source_id before
    invoking this.

    Raises ValueError on inconsistent grouping or empty input — both
    indicate a tier-7A bug rather than a normal data condition.
    """
    if not candidates:
        raise ValueError("classify_tier: empty candidates list")
    first = candidates[0]
    for c in candidates[1:]:
        if c.source_id != first.source_id:
            raise ValueError(
                f"classify_tier: inconsistent source_id {first.source_id!r} vs {c.source_id!r}"
            )

    n = len(candidates)
    votes = [
        ModelVote(
            model=c.model,
            citation_id_raw=c.citation_id,
            citation_id_normalized=normalize_citation_id(c.citation_id) if c.citation_id else "",
            target_mechanism=c.target_mechanism,
            mapping_justification=c.mapping_justification,
            parse_error=c.parse_error,
        )
        for c in candidates
    ]

    # Tier the agreement
    non_empty = [v.citation_id_normalized for v in votes if v.citation_id_normalized]
    empty_count = n - len(non_empty)

    # SALVAGE: model-side "no analog" consensus (>= n-1 empty answers)
    if empty_count >= n - 1:
        return _build(first, votes, "SALVAGE", "")

    counter = Counter(non_empty)
    top_id, top_count = counter.most_common(1)[0]

    if top_count == n:
        # HIGH: unanimous and non-empty
        return _build(first, votes, "HIGH", top_id)

    # Strict majority threshold: > n/2 (so 3/4, 4/5, 5/8, etc.)
    if top_count * 2 > n:
        return _build(first, votes, "MED", top_id)

    # No citation-level majority — check parent agreement
    parents = [_extract_parent(nid) for nid in non_empty]
    parent_counter = Counter(parents)
    top_parent, top_parent_count = parent_counter.most_common(1)[0]
    if top_parent_count == n:
        # MED-via-parent: all N agreed on parent article, split on sub-clause
        return _build(first, votes, "MED", top_parent)

    # LOW: dissensus
    return _build(first, votes, "LOW", top_id)


def _build(
    seed: EnsembleCandidate,
    votes: list[ModelVote],
    tier: Tier,
    consensus_citation_id: str,
) -> TieredPair:
    return TieredPair(
        source_id=seed.source_id,
        source_jurisdiction=seed.source_jurisdiction,
        source_framework=seed.source_framework,
        source_citation_id=seed.source_citation_id,
        source_mechanism=seed.source_mechanism,
        target_jurisdiction=seed.target_jurisdiction,
        target_framework=seed.target_framework,
        tier=tier,
        consensus_citation_id=consensus_citation_id,
        votes=votes,
    )


@validated
def tier_framework_pair(
    framework_pair: str,
    raw_dir: Path,
    expected_models: list[str] | None = None,
) -> list[TieredPair]:
    """Load all `data/ensemble/raw/{framework_pair}__{model}.jsonl` files,
    group by source_id, and tier each group.

    `expected_models`: if provided, fail loudly when a source_id is missing
    a vote from any expected model. Without this guard, a partial run could
    silently bias tiering toward whichever models did complete.
    """
    pattern = f"{framework_pair}__*.jsonl"
    files = sorted(raw_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No raw ensemble outputs found at {raw_dir}/{pattern}")

    by_source: dict[str, list[EnsembleCandidate]] = defaultdict(list)
    for path in files:
        for cand in _read_candidates(path):
            by_source[cand.source_id].append(cand)

    if expected_models is not None:
        expected_set = set(expected_models)
        for sid, cands in by_source.items():
            got = {c.model for c in cands}
            missing = expected_set - got
            if missing:
                raise ValueError(
                    f"source_id={sid!r} missing votes from models: {sorted(missing)}. "
                    f"Got {sorted(got)}; expected {sorted(expected_set)}."
                )

    return [classify_tier(by_source[sid]) for sid in sorted(by_source)]


def _read_candidates(path: Path) -> Iterable[EnsembleCandidate]:
    text = path.read_text(encoding="utf-8")
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            yield EnsembleCandidate.model_validate_json(line)
        except Exception as exc:
            raise ValueError(f"{path}:{lineno}: invalid EnsembleCandidate row: {exc}") from exc
