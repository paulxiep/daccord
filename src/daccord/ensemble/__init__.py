"""Ensemble candidate generation + HIGH/MED/LOW/SALVAGE tiering (tiers 6B + 7A).

Pipeline-spanning module: tier 7A writes `EnsembleCandidate` rows to disk,
tier 6B/8 reads them back and emits `TieredPair` rows for hand-validation.
See [docs/development_plan.md] tiers 6B + 7A + 8 for the gold-set pipeline.

## ⚠ Raw-data immutability rule (canonical declaration in `strategy.py`)

`data/ensemble/raw/{framework_pair}__{model_slug}.jsonl` rows are
**write-once per (file, source_id)** and immutable thereafter. The only
sanctioned mutation is `prune_parse_errors` (drops parse_error rows; never
touches successful rows) followed by `append_candidate` of the retry
result — used by `scripts/run_ensemble.py run-paid --retry-errors`.

All three mutators (`append_candidate`, `write_candidates_atomic`,
`prune_parse_errors`) enforce this at the code level via
`ImmutabilityViolation`. Future code that reads raw must produce its
output in separate files/directories — derivatives never edit raw.
"""

from daccord.ensemble.prompt import BatchPrompt, model_slug
from daccord.ensemble.schema import EnsembleCandidate, ModelVote, TieredPair
from daccord.ensemble.tier import classify_tier, tier_framework_pair

__all__ = [
    "BatchPrompt",
    "EnsembleCandidate",
    "ModelVote",
    "TieredPair",
    "classify_tier",
    "model_slug",
    "tier_framework_pair",
]
