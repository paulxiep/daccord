"""Gold-set artifacts — hand-validated cross-jurisdiction mapping pairs.

Pipeline-spanning module: tiers 2A/2B/7A/9/10A all import from here.
See [schema.py] for the canonical row + collection shapes.
"""

from daccord.gold.schema import GoldPair, GoldSet

__all__ = ["GoldPair", "GoldSet"]
