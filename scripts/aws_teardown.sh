#!/usr/bin/env bash
# Tier-6C teardown for D'accord M2 AWS resources.
#
# Removes:
#   1. S3 bucket `daccord-dev-{account_id}` AND all objects (including
#      versioned objects) inside it. Destructive.
#   2. IAM role `DaccordBedrockBatchService` + its inline policy.
#
# Does NOT touch:
#   - Bedrock model access (account-wide, leave granted)
#   - The existing $50/month budget (caravan-poc)
#   - The caravan-poc IAM user
#
# Run BEFORE deleting the AWS account or rotating profiles.
#
# Usage:  AWS_PROFILE=caravan-poc bash scripts/aws_teardown.sh
set -euo pipefail

PROFILE="${AWS_PROFILE:-caravan-poc}"
# Match the region used by aws_setup.sh (M2 Bedrock batch = us-east-1).
REGION="${AWS_REGION:-us-east-1}"
ROLE_NAME="DaccordBedrockBatchService"

ACCOUNT_ID=$(aws sts get-caller-identity --profile "$PROFILE" --query Account --output text)
BUCKET="daccord-dev-${ACCOUNT_ID}"

echo ">> Tearing down D'accord AWS resources in account $ACCOUNT_ID …"

# -------- 1. S3 bucket (delete all versions + bucket) ----------------------

if aws s3api head-bucket --bucket "$BUCKET" --profile "$PROFILE" 2>/dev/null; then
  echo ">> Deleting all object versions in s3://$BUCKET …"
  # Versioned-bucket deletion: list & delete all versions + delete markers.
  aws s3api list-object-versions \
    --bucket "$BUCKET" \
    --output json \
    --profile "$PROFILE" \
    --query '{Objects: [].{Key: Key, VersionId: VersionId}}' \
    | python -c "import sys, json; d = json.load(sys.stdin); print(json.dumps(d) if d.get('Objects') else '')" \
    > /tmp/daccord-versions.json || true

  if [ -s /tmp/daccord-versions.json ]; then
    aws s3api delete-objects \
      --bucket "$BUCKET" \
      --delete file:///tmp/daccord-versions.json \
      --profile "$PROFILE" >/dev/null
  fi

  aws s3api list-object-versions \
    --bucket "$BUCKET" \
    --output json \
    --profile "$PROFILE" \
    --query '{Objects: DeleteMarkers[].{Key: Key, VersionId: VersionId}}' \
    | python -c "import sys, json; d = json.load(sys.stdin); print(json.dumps(d) if d.get('Objects') else '')" \
    > /tmp/daccord-markers.json || true

  if [ -s /tmp/daccord-markers.json ]; then
    aws s3api delete-objects \
      --bucket "$BUCKET" \
      --delete file:///tmp/daccord-markers.json \
      --profile "$PROFILE" >/dev/null
  fi

  echo ">> Deleting bucket s3://$BUCKET …"
  aws s3api delete-bucket --bucket "$BUCKET" --region "$REGION" --profile "$PROFILE"
else
  echo "   bucket s3://$BUCKET not found — skipping"
fi

# -------- 2. IAM role ------------------------------------------------------

if aws iam get-role --role-name "$ROLE_NAME" --profile "$PROFILE" >/dev/null 2>&1; then
  echo ">> Removing inline policy from role $ROLE_NAME …"
  aws iam delete-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-name "DaccordS3BatchIO" \
    --profile "$PROFILE" 2>/dev/null || true

  echo ">> Deleting IAM role $ROLE_NAME …"
  aws iam delete-role --role-name "$ROLE_NAME" --profile "$PROFILE"
else
  echo "   role $ROLE_NAME not found — skipping"
fi

echo ""
echo "================ D'accord AWS teardown complete ================"
echo "  Bucket s3://$BUCKET removed (if existed)"
echo "  Role $ROLE_NAME removed (if existed)"
echo "  Caravan-poc profile + budget untouched"
