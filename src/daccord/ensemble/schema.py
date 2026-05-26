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
    """One source clause + its tier label from ensemble agreement.

    `consensus_citation_id` is the normalized citation_id that the tier
    label refers to:
      - HIGH: the unanimous answer
      - MED: the majority answer (3/4) OR the shared parent if all 4
        agreed on the parent article but split on sub-clauses
      - LOW: the most-voted answer (may be a plurality); included for
        downstream auditing
      - SALVAGE: empty string (no usable citation across the ensemble)
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
    votes: list[ModelVote]
