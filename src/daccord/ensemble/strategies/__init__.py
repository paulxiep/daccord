"""Concrete `EnsembleStrategy` implementations.

Each module here is ONE tier-7A execution path documented in
[docs/7a_path.md](../../../../docs/7a_path.md):

  - `bedrock.py`  — Path 1: AWS Bedrock batch + sync invocation
  - `paid_api.py` — Path 2: paid direct API ensemble (Anthropic + OpenAI +
                    Google + Together)
  - (future)      — Path 3: 4-provider free-tier API ensemble (NVIDIA NIM,
                    Mistral, OpenRouter, Gemini free)

All strategies share the resilience contract from
`daccord.ensemble.strategy` so any partial run resumes by re-invocation.
"""

from daccord.ensemble.strategies.bedrock import (
    BedrockBatchStrategy,
    BedrockSyncStrategy,
)
from daccord.ensemble.strategies.paid_api import PaidAPIStrategy

__all__ = [
    "BedrockBatchStrategy",
    "BedrockSyncStrategy",
    "PaidAPIStrategy",
]
