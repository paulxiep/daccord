"""Ensemble candidate generation + HIGH/MED/LOW/SALVAGE tiering (tiers 6B + 7A).

Pipeline-spanning module: tier 7A writes `EnsembleCandidate` rows to disk,
tier 6B/8 reads them back and emits `TieredPair` rows for hand-validation.
See [docs/development_plan.md] tiers 6B + 7A + 8 for the gold-set pipeline.
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
