"""Tier 6C / 7A — AWS resource conventions for the M2 Bedrock-batch ensemble.

Single source of truth for:
  - the F9-E ensemble model IDs (4 seats, all Bedrock-hosted)
  - the AWS region for batch inference (us-east-1 — only region with Llama 4)
  - the S3 bucket name pattern (`daccord-dev-{account_id}`)
  - the Bedrock-batch service role + inline policy names
  - the `Project=daccord` tag value
  - profile resolution (env var > aws-account.yaml > literal fallback)

Consumed by:
  - `scripts/check_aws_setup.py` (tier 6C verification)
  - tier 7A `BedrockBatchClient` + `scripts/run_ensemble.py` (next session)
  - any future M2-touching code (tier 8 hand-validation tooling, etc.)

Changing any constant here propagates everywhere. Pair with edits to
`scripts/aws_setup.{sh,ps1}` and `scripts/aws_teardown.{sh,ps1}` — those
are standalone scripts that grep `aws-account.yaml` directly rather than
importing this module (they run on the host, outside the docker venv).
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from daccord.validation import validated

# -------- M2 Bedrock-batch resource conventions --------

REGION = "us-east-1"
"""Batch-inference region. us-east-1 is the only Bedrock region that exposes
Llama 4 Scout / Maverick (US Geo CRIS, no Global CRIS as of 2026-05). M5
SageMaker stand-up uses ap-southeast-1 separately (low RTT for live demo)."""

PROJECT_TAG = "daccord"
"""AWS resource tag key=`Project` value=this. Used by the existing $50/month
Caravan-PoC budget to attribute D'accord spend separately."""

ROLE_NAME = "DaccordBedrockBatchService"
"""IAM service role that Bedrock assumes during batch jobs to read prompt
JSONLs from + write output JSONLs to the daccord S3 bucket."""

INLINE_POLICY_NAME = "DaccordS3BatchIO"
"""Name of the inline policy attached to `ROLE_NAME` that grants
s3:Get/Put/List on the daccord bucket only."""

F9_BEDROCK_MODELS: list[str] = [
    "meta.llama4-scout-17b-instruct-v1:0",
    "meta.llama4-maverick-17b-instruct-v1:0",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "amazon.nova-2-lite-v1:0",
]
"""Committed F9-E ensemble seats. Order is stable — tier 7A iteration +
tier 6B classifier output rely on the same model-ID strings here matching
the keys in `costs/config.toml [pricing.bedrock_batch.*]`."""

# -------- Derived helpers (account-ID dependent) --------


@validated
def bucket_name(account_id: str) -> str:
    """S3 bucket holding batch input + output JSONLs.

    `account_id` is the 12-digit string from `aws sts get-caller-identity`;
    we don't fetch it here to keep this module side-effect-free (importable
    in tests without a live AWS session)."""
    return f"daccord-dev-{account_id}"


@validated
def bucket_arn(account_id: str) -> str:
    return f"arn:aws:s3:::{bucket_name(account_id)}"


@validated
def role_arn(account_id: str) -> str:
    """ARN passed to `bedrock:CreateModelInvocationJob` as `roleArn`."""
    return f"arn:aws:iam::{account_id}:role/{ROLE_NAME}"


@validated
def s3_input_prefix(account_id: str, framework_pair: str, model: str) -> str:
    """Tier 7A submit-phase upload destination for one (pair, model) batch.

    Convention: `s3://{bucket}/ensemble/in/{framework_pair}__{model}.jsonl`.
    Matches the on-disk landing pattern at `data/ensemble/raw/{pair}__{model}.jsonl`
    so the same naming logic works on both sides of the cloud round-trip."""
    return f"s3://{bucket_name(account_id)}/ensemble/in/{framework_pair}__{model}.jsonl"


@validated
def s3_output_prefix(account_id: str, framework_pair: str, model: str) -> str:
    """Tier 7A poll-phase download source for one (pair, model) batch."""
    return f"s3://{bucket_name(account_id)}/ensemble/out/{framework_pair}__{model}/"


# -------- Profile resolution: env > aws-account.yaml > literal fallback --------

_FALLBACK_PROFILE = "caravan-poc"
"""Last-resort profile if neither $AWS_PROFILE nor aws-account.yaml resolves."""


def _repo_root() -> Path:
    """Repo root located 3 parents up from this file
    (src/daccord/aws/m2.py → src/daccord/aws → src/daccord → src → root)."""
    return Path(__file__).resolve().parents[3]


def _profile_from_yaml(yaml_path: Path) -> str | None:
    if not yaml_path.exists():
        return None
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        return data.get("profile") if isinstance(data, dict) else None
    except yaml.YAMLError:
        return None


@validated
def resolve_profile(arg_profile: str | None = None) -> str:
    """Profile precedence: --profile arg > AWS_PROFILE env > aws-account.yaml > fallback.

    The `caravan-poc` fallback is committed for repo-shared reproducibility;
    individual users override via `aws-account.yaml` (gitignored) or env."""
    if arg_profile:
        return arg_profile
    env_profile = os.environ.get("AWS_PROFILE")
    if env_profile:
        return env_profile
    yaml_profile = _profile_from_yaml(_repo_root() / "aws-account.yaml")
    if yaml_profile:
        return yaml_profile
    return _FALLBACK_PROFILE
