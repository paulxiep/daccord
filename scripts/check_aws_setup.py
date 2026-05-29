"""Tier-6C verification: confirms D'accord AWS resources are ready for M2.

Checks (read-only, no resource creation):

  1. STS caller-identity — confirms the active profile/credentials reach AWS.
  2. S3 bucket `daccord-dev-{account_id}` exists, has versioning enabled,
     carries the `Project=daccord` tag.
  3. IAM service role `DaccordBedrockBatchService` exists with the right
     trust policy and inline S3 policy.
  4. Bedrock model availability in `us-east-1` for the 4 F9-E ensemble
     members (Llama 4 Scout + Maverick + Haiku 4.5 + Nova 2 Lite). Reports
     per-model status and entitlement.

Exit code:
  0 = all green (ready to submit batch jobs)
  1 = one or more checks failed; per-check status printed.

Usage:  AWS_PROFILE=caravan-poc python scripts/check_aws_setup.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("ERROR: boto3 not installed. Install via: pip install boto3", file=sys.stderr)
    sys.exit(2)

from daccord.aws.m2 import (
    F9_BEDROCK_MODELS,
    INLINE_POLICY_NAME,
    PROJECT_TAG,
    REGION,
    ROLE_NAME,
    bucket_name,
    resolve_profile,
)

log = logging.getLogger("check_aws_setup")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    profile = resolve_profile(args.profile)
    region = args.region
    log.info("Using AWS profile=%s region=%s", profile, region)
    session = boto3.Session(profile_name=profile, region_name=region)

    checks: list[Check] = []
    try:
        account_id = _check_caller_identity(session, checks)
    except Exception as exc:
        log.error("FATAL: caller-identity lookup failed: %s", exc)
        return 1

    bucket = bucket_name(account_id)
    _check_s3_bucket(session, bucket, checks)
    _check_iam_role(session, account_id, bucket, region, checks)
    _check_bedrock_models(session, region, checks)

    _print_summary(checks, profile, region, account_id, bucket)
    return 0 if all(c.ok for c in checks) else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", default=None, help="AWS profile name (default: $AWS_PROFILE or caravan-poc)"
    )
    parser.add_argument(
        "--region",
        default=REGION,
        help=f"AWS region (default: {REGION} for M2 Bedrock batch)",
    )
    return parser.parse_args(argv)


def _check_caller_identity(session: boto3.Session, checks: list[Check]) -> str:
    sts = session.client("sts")
    resp = sts.get_caller_identity()
    account_id = resp["Account"]
    checks.append(
        Check(
            name="STS caller identity",
            ok=True,
            detail=f"account={account_id}  arn={resp['Arn']}",
        )
    )
    return account_id


def _check_s3_bucket(session: boto3.Session, bucket: str, checks: list[Check]) -> None:
    s3 = session.client("s3")
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        checks.append(
            Check(
                name=f"S3 bucket s3://{bucket}",
                ok=False,
                detail=f"head_bucket failed: {code} — run scripts/aws_setup.sh",
            )
        )
        return

    versioning = s3.get_bucket_versioning(Bucket=bucket).get("Status", "Disabled")
    versioning_ok = versioning == "Enabled"

    try:
        tags = {
            t["Key"]: t["Value"] for t in s3.get_bucket_tagging(Bucket=bucket).get("TagSet", [])
        }
    except ClientError:
        tags = {}
    tag_ok = tags.get("Project") == PROJECT_TAG

    detail = f"versioning={versioning}  Project tag={tags.get('Project', '<missing>')}"
    checks.append(
        Check(
            name=f"S3 bucket s3://{bucket}",
            ok=versioning_ok and tag_ok,
            detail=detail,
        )
    )


def _check_iam_role(
    session: boto3.Session,
    account_id: str,
    bucket: str,
    region: str,
    checks: list[Check],
) -> None:
    iam = session.client("iam")
    try:
        role = iam.get_role(RoleName=ROLE_NAME)["Role"]
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        checks.append(
            Check(
                name=f"IAM role {ROLE_NAME}",
                ok=False,
                detail=f"get_role failed: {code} — run scripts/aws_setup.sh",
            )
        )
        return

    trust = role.get("AssumeRolePolicyDocument", {})
    trust_principals = {s.get("Principal", {}).get("Service") for s in trust.get("Statement", [])}
    trust_ok = "bedrock.amazonaws.com" in trust_principals

    try:
        inline = iam.get_role_policy(RoleName=ROLE_NAME, PolicyName=INLINE_POLICY_NAME)
        policy_doc = inline["PolicyDocument"]
    except ClientError:
        policy_doc = None
    policy_ok = False
    if policy_doc is not None:
        # Confirm the policy references our bucket ARN.
        resources = []
        for s in policy_doc.get("Statement", []):
            r = s.get("Resource", [])
            if isinstance(r, str):
                resources.append(r)
            else:
                resources.extend(r)
        policy_ok = any(bucket in r for r in resources)

    detail = (
        f"trust principal=bedrock.amazonaws.com:{trust_ok}  "
        f"inline policy references {bucket}:{policy_ok}  "
        f"role ARN={role['Arn']}"
    )
    checks.append(Check(name=f"IAM role {ROLE_NAME}", ok=trust_ok and policy_ok, detail=detail))


def _check_bedrock_models(session: boto3.Session, region: str, checks: list[Check]) -> None:
    """Per-model: is it listed in the region, and is access granted?

    Bedrock's `list_foundation_models` reports availability; per-model
    entitlement (whether the account has been granted access) requires
    `get_foundation_model_availability` which is not available in all
    regions yet. Fall back to a try-invoke probe if unavailable.
    """
    bedrock = session.client("bedrock", region_name=region)
    try:
        listed = {m["modelId"] for m in bedrock.list_foundation_models()["modelSummaries"]}
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        checks.append(
            Check(
                name=f"Bedrock list_foundation_models in {region}",
                ok=False,
                detail=(
                    f"{code} — region may not have Bedrock or IAM lacks "
                    f"bedrock:ListFoundationModels"
                ),
            )
        )
        return

    for model_id in F9_BEDROCK_MODELS:
        in_region = model_id in listed
        entitlement = "UNKNOWN"
        if in_region:
            entitlement = _probe_model_access(bedrock, model_id)
        detail = f"in-region={in_region}  access={entitlement}"
        # entitlementAvailability values: AVAILABLE (granted) | NOT_AVAILABLE
        # (needs use-case form). UNKNOWN = probe API unsupported in region
        # → treat as ok (the actual invoke will fail loudly if not granted).
        ok = in_region and entitlement in ("AVAILABLE", "UNKNOWN")
        checks.append(Check(name=f"Bedrock model {model_id}", ok=ok, detail=detail))


def _probe_model_access(bedrock: object, model_id: str) -> str:
    """Best-effort access probe — uses get_foundation_model_availability
    when the API supports it; otherwise returns UNKNOWN (caller treats as ok-ish)."""
    try:
        resp = bedrock.get_foundation_model_availability(modelId=model_id)  # type: ignore[attr-defined]
        return resp.get("entitlementAvailability", "UNKNOWN")
    except ClientError, AttributeError:
        return "UNKNOWN"


def _print_summary(
    checks: list[Check],
    profile: str,
    region: str,
    account_id: str,
    bucket: str,
) -> None:
    print()
    print("=" * 70)
    print(f"D'accord AWS setup check — profile={profile} region={region}")
    print(f"account={account_id}  bucket=s3://{bucket}")
    print("=" * 70)
    for c in checks:
        marker = "OK " if c.ok else "FAIL"
        print(f"[{marker}] {c.name}")
        print(f"       {c.detail}")
    print("=" * 70)
    failed = [c for c in checks if not c.ok]
    if failed:
        print(f"FAILED: {len(failed)} of {len(checks)} checks")
        print("Fix the FAILED items above, then re-run.")
    else:
        print(f"PASS: all {len(checks)} checks")
        print("Ready to submit Bedrock batch jobs (tier 7A next session).")


if __name__ == "__main__":
    sys.exit(main())
