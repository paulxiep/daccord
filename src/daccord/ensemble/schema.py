"""Ensemble data shapes (tier 7A output → tier 6B/8 input → tier 9 gold).

Three shapes:

  - `EnsembleCandidate` — one model's candidate for one source clause.
    Tier 7A writes a JSONL of these per `(framework_pair, model)` to
    `data/ensemble/raw/{framework_pair}__{model}.jsonl`.
  - `ModelVote` — per-model row inside a tiered result; preserves the raw
    citation_id + the normalized form used for agreement counting.
  - `TieredPair` — one source clause + its HIGH/MED/LOW/SALVAGE label after
    4-way agreement scoring. Tier 6B writes JSONL of these to
    `data/ensemble/tiered/{framework_pair}.jsonl`; tier 8 hand-validates
    MED/LOW/SALVAGE; tier 9 freezes HIGH + validated MED into
    `data/gold/gold_v1.jsonl`.
"""

from __future__ import annotations

from typing import Literal

from daccord.validation import ValidatedModel

Tier = Literal["HIGH", "MED", "LOW", "SALVAGE"]
TIERS: tuple[Tier, ...] = ("HIGH", "MED", "LOW", "SALVAGE")

# Model-id prefixes that identify "local" (non-LLM) seats. Their votes are
# preserved in `TieredPair.votes` for downstream consumption but are NOT
# counted in the LLM-consensus math (agreement_score, tier). The classic
# example is the tier-6B++ RAG seat (`local-rag-mpnet`) — semantic-similarity
# retrieval is informative but a different epistemic basis than LLM
# reasoning, so it shouldn't downgrade a 4/4 LLM unanimous HIGH.
LOCAL_SEAT_MODEL_PREFIXES: tuple[str, ...] = ("local-",)


def is_local_seat(model: str) -> bool:
    """True iff `model` is a local-compute seat (vs paid-API LLM seat).

    Used by `classify_tier` to split votes into LLM-consensus voters vs
    side-info voters. Centralised here so future local seats (e.g. local
    HF Qwen) automatically follow the same rule.
    """
    return any(model.startswith(p) for p in LOCAL_SEAT_MODEL_PREFIXES)


class EnsembleCandidate(ValidatedModel):
    """One model's candidate mapping for one source clause.

    The `source_*` fields identify the query; `model` identifies which
    ensemble seat produced this candidate; the `citation_id` /
    `target_mechanism` / `mapping_justification` are the model's answer
    (matching `CitationCandidate` from the eval schema).

    `parse_error` is non-None when the model's raw text could not be parsed
    into the structured shape (e.g. local HF returned malformed JSON). The
    tiering script treats parse errors as missing votes — they reduce the
    agreement count but don't crash the run.
    """

    source_id: str
    source_jurisdiction: str
    source_framework: str
    source_citation_id: str
    source_mechanism: str
    target_jurisdiction: str
    target_framework: str
    model: str
    citation_id: str
    target_mechanism: str
    mapping_justification: str
    parse_error: str | None = None


class ModelVote(ValidatedModel):
    """One model's contribution to a tiered pair — raw + normalized citation."""

    model: str
    citation_id_raw: str
    citation_id_normalized: str
    target_mechanism: str
    mapping_justification: str
    parse_error: str | None = None


class TieredPair(ValidatedModel):
    """One source clause + its ensemble-agreement scoring.

    Three fuzzy fields are the primary signal; `tier` is a derived legacy
    bucket kept for back-compat with tier-6B-era consumers (splits CLI,
    early reporting). New consumers should prefer `agreement_score` +
    `valid_vote_count` directly.

    Fuzzy fields:
      - `valid_vote_count`: number of seats that produced a parseable,
        non-empty citation_id (∈ [0, N]).
      - `consensus_vote_count`: how many valid votes agreed on
        `consensus_citation_id` (after normalization, with parent-fallback
        when no citation-level majority exists — see tier.py).
      - `agreement_score`: `consensus_vote_count / valid_vote_count` ∈ [0, 1],
        OR `-1.0` sentinel when `valid_vote_count == 0` (SALVAGE).

    Legacy `tier` bucket mapping (computed from the fuzzy fields):
      | agreement_score | valid_vote_count | tier    |
      |-----------------|------------------|---------|
      | 1.0             | ≥ 2              | HIGH    |
      | ≥ 0.6           | ≥ 2              | MED     |
      | < 0.6           | ≥ 1              | LOW     |
      | (any)           | 0                | SALVAGE |

    `consensus_citation_id` is the normalized citation_id (or parent
    article) that the agreement_score refers to. Empty when SALVAGE.
    """

    source_id: str
    source_jurisdiction: str
    source_framework: str
    source_citation_id: str
    source_mechanism: str
    target_jurisdiction: str
    target_framework: str
    tier: Tier
    consensus_citation_id: str
    valid_vote_count: int
    consensus_vote_count: int
    agreement_score: float
    votes: list[ModelVote]
    # Local-seat side-info: populated when at least one local-prefixed
    # vote (e.g. `local-rag-mpnet`) participates. `rag_concurs` is True
    # iff the local seat's normalized citation_id matches
    # `consensus_citation_id` (both non-empty). `rag_vote_citation_id`
    # carries the local seat's normalized vote for downstream display.
    # These are derived fields; downstream consumers (splits + labeler)
    # can promote MED rows where `rag_concurs=True` as a separate
    # opt-in path. Default False/"" so older `TieredPair` JSONL parses
    # without these fields stays valid.
    rag_concurs: bool = False
    rag_vote_citation_id: str = ""
