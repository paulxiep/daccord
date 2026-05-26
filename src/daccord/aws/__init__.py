"""D'accord AWS resource conventions — shared across tier 6C verification,
tier 7A Bedrock batch, and (later) tier 14 SageMaker stand-up.

Submodules:
  - `m2`   — M2 Bedrock-batch constants (region, bucket pattern, role name,
            F9 ensemble model IDs) + profile resolution helper.

  (tier 14 → `m5` submodule expected here.)
"""

from daccord.aws import m2

__all__ = ["m2"]
