"""Tier-6B fuzzy-agreement classifier (with HIGH/MED/LOW/SALVAGE back-compat).

Reads ensemble candidates grouped by source clause, computes three fuzzy
agreement fields, and derives a legacy HIGH/MED/LOW/SALVAGE bucket for
tier-6B-era consumers (splits CLI, early reporting).

Primary fields (new):
  - `valid_vote_count`  — seats with a parseable, non-empty citation_id.
  - `consensus_vote_count` — how many valid votes agreed on `consensus_citation_id`
    (after normalization, with parent-fallback when no citation-level majority
    exists).
  - `agreement_score` — `consensus_vote_count / valid_vote_count`, in [0, 1],
    OR the sentinel `-1.0` when `valid_vote_count == 0` (SALVAGE).

Derivation order:
  1. **Full citation-level agreement** (every valid vote picked the same
     normalized citation_id) → `agreement_score = 1.0`, consensus is that ID.
  2. **Citation-level strict majority** (top_count > valid_vote_count / 2)
     → `agreement_score = top_count / valid_vote_count`, consensus is the
     majority ID. Parent-fallback is NOT used here: the more specific
     sub-citation is a stronger signal than the parent.
  3. **Unanimous parent agreement** (all valid votes share the same parent
     article — sub-clause split) → `agreement_score = 1.0`, consensus is
     the parent string (e.g. "32" when votes were "32(1)", "32(1)(a)",
     "32(2)"). Informative for the hand-val reviewer.
  4. **Plurality only** → `agreement_score = top_count / valid_vote_count`,
     consensus is the plurality ID. Typically derives LOW.
  5. **All votes empty or parse_error** → SALVAGE, sentinel score `-1.0`.

Legacy `tier` bucket (derived from the fuzzy fields):
  | agreement_score | valid_vote_count | tier    |
  |-----------------|------------------|---------|
  | 1.0             | ≥ 2              | HIGH    |
  | ≥ 0.6           | ≥ 2              | MED     |
  | < 0.6           | ≥ 1              | LOW     |
  | (any)           | 0                | SALVAGE |

Parent extraction: registry IDs like "32(1)(a)" → "32"; "5A" → "5A"; "มาตรา 19"
(Thai) → "มาตรา 19". Splits only on "(", not language-aware — sub-clause-split
is mostly a Latin-script (EU/UK) phenomenon.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from daccord.ensemble.schema import (
    EnsembleCandidate,
    ModelVote,
    Tier,
    TieredPair,
    is_local_seat,
)
from daccord.eval.scoring import normalize_citation_id
from daccord.validation import validated

_PARENT_SPLIT = re.compile(r"^([^(]+?)\s*(?:\(.*)?$")


def _extract_parent(normalized_id: str) -> str:
    if not normalized_id:
        return ""
    m = _PARENT_SPLIT.match(normalized_id)
    return m.group(1).strip() if m else normalized_id


def _derive_tier(agreement_score: float, valid_vote_count: int) -> Tier:
    if valid_vote_count == 0:
        return "SALVAGE"
    if agreement_score >= 1.0 and valid_vote_count >= 2:
        return "HIGH"
    if agreement_score >= 0.6 and valid_vote_count >= 2:
        return "MED"
    return "LOW"


@validated
def classify_tier(candidates: list[EnsembleCandidate]) -> TieredPair:
    """Score one source clause's N candidates → TieredPair (fuzzy, LLM-only).

    All candidates MUST share the same source_id, source_*, and target_*
    fields. The caller (`tier_framework_pair`) groups by source_id before
    invoking this.

    **LLM-only consensus**: `agreement_score`, `valid_vote_count`, and the
    derived `tier` are computed over LLM seats only (votes whose model
    does NOT start with a `LOCAL_SEAT_MODEL_PREFIXES` prefix — see
    `daccord.ensemble.schema.is_local_seat`). Local seats (currently:
    the tier-6B++ RAG seat `local-rag-mpnet`) are preserved in the
    `votes` list for downstream consumers but treated as side-info — they
    don't downgrade a 4/4 LLM unanimous HIGH because retrieval and LLM
    reasoning are different epistemic bases.

    The local seat's vote is surfaced via two derived fields:
      - `rag_concurs: bool` — True iff a local seat exists AND its
        normalized citation_id matches the (non-empty) LLM consensus.
      - `rag_vote_citation_id: str` — the local seat's normalized vote
        (empty if no local seat ran or if it returned empty).

    Splits + labeler can use these to opt-in promote MED rows where the
    independent retrieval signal concurs with the LLM consensus.

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

    # Split LLM-consensus voters from local-seat side-info voters.
    llm_votes = [v for v in votes if not is_local_seat(v.model)]
    local_votes = [v for v in votes if is_local_seat(v.model)]

    # Local seat's vote is captured separately for the rag_concurs derivation.
    # We use only the first local-seat vote — there's a single RAG seat per
    # source clause today; multi-local-seat support would need a policy.
    rag_vote_citation_id = local_votes[0].citation_id_normalized if local_votes else ""

    valid_norm = [v.citation_id_normalized for v in llm_votes if v.citation_id_normalized]
    valid_vote_count = len(valid_norm)

    if valid_vote_count == 0:
        return _build(
            first,
            votes,
            tier="SALVAGE",
            consensus_citation_id="",
            valid_vote_count=0,
            consensus_vote_count=0,
            agreement_score=-1.0,
            rag_concurs=False,
            rag_vote_citation_id=rag_vote_citation_id,
        )

    citation_counter = Counter(valid_norm)
    top_id, top_count = citation_counter.most_common(1)[0]

    consensus_citation_id: str
    consensus_vote_count: int
    agreement_score: float

    if top_count == valid_vote_count:
        consensus_citation_id = top_id
        consensus_vote_count = top_count
        agreement_score = 1.0
    elif top_count * 2 > valid_vote_count:
        consensus_citation_id = top_id
        consensus_vote_count = top_count
        agreement_score = top_count / valid_vote_count
    else:
        # No citation-level majority — try unanimous parent fallback.
        # When all valid votes share a parent article (sub-clause split),
        # surface the parent as the consensus. The `top_parent_count >
        # top_count` check filters the degenerate case where parent and
        # citation agree at the same count (no real promotion happens).
        parents = [_extract_parent(nid) for nid in valid_norm]
        parent_counter = Counter(parents)
        top_parent, top_parent_count = parent_counter.most_common(1)[0]
        if top_parent_count == valid_vote_count and top_parent_count > top_count:
            consensus_citation_id = top_parent
            consensus_vote_count = top_parent_count
            agreement_score = 1.0
        else:
            consensus_citation_id = top_id
            consensus_vote_count = top_count
            agreement_score = top_count / valid_vote_count

    tier = _derive_tier(agreement_score, valid_vote_count)
    # rag_concurs: True iff a local seat voted AND its citation matches the
    # consensus. Concurrence on parent-fallback consensus counts too.
    rag_concurs = bool(
        rag_vote_citation_id
        and consensus_citation_id
        and rag_vote_citation_id == consensus_citation_id
    )
    return _build(
        first,
        votes,
        tier=tier,
        consensus_citation_id=consensus_citation_id,
        valid_vote_count=valid_vote_count,
        consensus_vote_count=consensus_vote_count,
        agreement_score=agreement_score,
        rag_concurs=rag_concurs,
        rag_vote_citation_id=rag_vote_citation_id,
    )


def _build(
    seed: EnsembleCandidate,
    votes: list[ModelVote],
    *,
    tier: Tier,
    consensus_citation_id: str,
    valid_vote_count: int,
    consensus_vote_count: int,
    agreement_score: float,
    rag_concurs: bool = False,
    rag_vote_citation_id: str = "",
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
        valid_vote_count=valid_vote_count,
        consensus_vote_count=consensus_vote_count,
        agreement_score=agreement_score,
        votes=votes,
        rag_concurs=rag_concurs,
        rag_vote_citation_id=rag_vote_citation_id,
    )


@validated
def tier_framework_pair(
    framework_pair: str,
    raw_dir: Path,
    expected_models: list[str] | None = None,
    extra_dirs: list[Path] | None = None,
) -> list[TieredPair]:
    """Load all `{raw_dir}/{framework_pair}__{model}.jsonl` (plus any
    `{extra_dir}/{framework_pair}__*.jsonl` files), group by source_id, and
    tier each group.

    `raw_dir` is the canonical immutable paid-API output (`data/ensemble/raw/`).
    `extra_dirs` lets the caller union in additional candidate sources —
    typically `[data/ensemble/raw_local/]` for the tier-6B++ RAG seat,
    which produces EnsembleCandidate-shape rows from local compute. Files
    from extra_dirs share the same `{pair}__{model}.jsonl` naming so the
    glob logic stays uniform. Models from extra_dirs vote alongside paid
    seats with equal weight in `classify_tier`'s fuzzy scoring.

    `expected_models`: if provided, fail loudly when a source_id is missing
    a vote from any expected model. Without this guard, a partial run could
    silently bias tiering toward whichever models did complete.
    """
    pattern = f"{framework_pair}__*.jsonl"
    files = sorted(raw_dir.glob(pattern))
    if extra_dirs is not None:
        for extra in extra_dirs:
            if extra.exists():
                files.extend(sorted(extra.glob(pattern)))
    if not files:
        raise FileNotFoundError(
            f"No raw ensemble outputs found at {raw_dir}/{pattern}"
            + (f" (also checked {[str(d) for d in (extra_dirs or [])]})" if extra_dirs else "")
        )

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
