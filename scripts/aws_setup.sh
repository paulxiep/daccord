#!/usr/bin/env bash
# Tier-6C AWS resource creation for D'accord M2 ensemble (Bedrock batch).
#
# Idempotent — re-running on an existing setup is a no-op (each command
# tolerates "already exists" responses).
#
# What this creates:
#   1. S3 bucket `daccord-dev-{account_id}` with versioning + Project=daccord tag
#      (reuses the existing $50/month budget by tag-association).
#   2. IAM service role `DaccordBedrockBatchService` that Bedrock assumes when
#      running batch inference jobs (reads/writes S3 input/output JSONL).
#
# What this does NOT create:
#   - IAM user / access keys — reuses your existing `caravan-poc` admin profile.
#   - AWS Budgets alarm — reuses the existing $50/month budget (caravan-poc).
#   - Bedrock model-access approvals — those go through the console form.
#     Run `python scripts/check_aws_setup.py` after this to see which models
#     are ACTIVE and which still need a use-case form.
#
# Teardown: `scripts/aws_teardown.sh` removes the bucket + role.
#
# Usage:  AWS_PROFILE=caravan-poc bash scripts/aws_setup.sh
set -euo pipefail

PROFILE="${AWS_PROFILE:-caravan-poc}"
# us-east-1 for M2 Bedrock batch (only region with Llama 4 access).
# M5 SageMaker stand-up later will use ap-southeast-1 separately.
REGION="${AWS_REGION:-us-east-1}"
PROJECT_TAG="daccord"
ROLE_NAME="DaccordBedrockBatchService"

echo ">> Resolving account ID via profile=$PROFILE …"
ACCOUNT_ID=$(aws sts get-caller-identity --profile "$PROFILE" --query Account --output text)
BUCKET="daccord-dev-${ACCOUNT_ID}"
echo "   account_id=$ACCOUNT_ID  bucket=$BUCKET  region=$REGION"

# -------- 1. S3 bucket ----------------------------------------------------

echo ">> Creating S3 bucket s3://$BUCKET (idempotent) …"
if aws s3api head-bucket --bucket "$BUCKET" --profile "$PROFILE" 2>/dev/null; then
  echo "   bucket already exists — skipping create"
else
  aws s3api create-bucket \
    --bucket "$BUCKET" \
    --region "$REGION" \
    --create-bucket-configuration "LocationConstraint=$REGION" \
    --profile "$PROFILE"
  echo "   bucket created"
fi

echo ">> Enabling versioning on s3://$BUCKET …"
aws s3api put-bucket-versioning \
  --bucket "$BUCKET" \
  --versioning-configuration Status=Enabled \
  --profile "$PROFILE"

echo ">> Tagging bucket Project=$PROJECT_TAG …"
aws s3api put-bucket-tagging \
  --bucket "$BUCKET" \
  --tagging "TagSet=[{Key=Project,Value=$PROJECT_TAG}]" \
  --profile "$PROFILE"

# -------- 2. Bedrock batch service role -----------------------------------

TRUST_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "bedrock.amazonaws.com" },
      "Action": "sts:AssumeRole",
      "Condition": {
        "StringEquals": { "aws:SourceAccount": "$ACCOUNT_ID" },
        "ArnLike": {
          "aws:SourceArn": "arn:aws:bedrock:$REGION:$ACCOUNT_ID:model-invocation-job/*"
        }
      }
    }
  ]
}
EOF
)

INLINE_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::$BUCKET/*"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::$BUCKET"
    }
  ]
}
EOF
)

echo ">> Creating IAM role $ROLE_NAME (idempotent) …"
if aws iam get-role --role-name "$ROLE_NAME" --profile "$PROFILE" >/dev/null 2>&1; then
  echo "   role already exists — updating trust policy"
  aws iam update-assume-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-document "$TRUST_POLICY" \
    --profile "$PROFILE"
else
  aws iam create-role \
    --role-name "$ROLE_NAME" \
    --assume-role-policy-document "$TRUST_POLICY" \
    --description "Assumed by Bedrock to read/write D'accord S3 batch I/O" \
    --tags "Key=Project,Value=$PROJECT_TAG" \
    --profile "$PROFILE"
  echo "   role created"
fi

echo ">> Attaching inline S3 policy to role …"
aws iam put-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name "DaccordS3BatchIO" \
  --policy-document "$INLINE_POLICY" \
  --profile "$PROFILE"

ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --profile "$PROFILE" --query 'Role.Arn' --output text)

echo ""
echo "================ D'accord AWS setup complete ================"
echo "  Profile     : $PROFILE"
echo "  Region      : $REGION"
echo "  S3 bucket   : s3://$BUCKET (versioning ON, Project=$PROJECT_TAG)"
echo "  Service role: $ROLE_ARN"
echo ""
echo "Next steps:"
echo "  1. Run 'python scripts/check_aws_setup.py' to verify + see"
echo "     which Bedrock models still need use-case forms in the console."
echo "  2. Open AWS Console → Bedrock → us-east-1 → Model access:"
echo "     submit the use-case form for any models reported as not granted."
echo "     Typically: Claude Haiku 4.5 (auto-approved <5 min); Llama 4 Scout/Maverick"
echo "     and Nova 2 Lite are usually instant-grant."
echo "  3. Re-run scripts/check_aws_setup.py until all 4 F9 models report access OK."
