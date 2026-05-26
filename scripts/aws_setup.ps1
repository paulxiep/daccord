# Tier-6C AWS resource creation for D'accord M2 ensemble (Bedrock batch).
#
# PowerShell port of aws_setup.sh for Windows users.
# Idempotent — re-running on an existing setup is a no-op.
#
# What this creates:
#   1. S3 bucket `daccord-dev-{account_id}` in us-east-1 with versioning +
#      Project=daccord tag.
#   2. IAM service role `DaccordBedrockBatchService` that Bedrock assumes
#      when running batch inference jobs.
#
# Region: us-east-1 for M2 Bedrock batch (only region with Llama 4 access).
# M5 SageMaker stand-up later will use ap-southeast-1 separately.
#
# Profile resolution order (highest priority first):
#   1. $env:AWS_PROFILE
#   2. `profile:` field in aws-account.yaml at repo root
#   3. literal fallback "caravan-poc"
#
# Usage:
#   .\scripts\aws_setup.ps1
#   # or with a non-default profile:
#   $env:AWS_PROFILE = "your-profile"; .\scripts\aws_setup.ps1

$ErrorActionPreference = "Stop"

# Disable AWS CLI's interactive pager (defaults to `more` on Windows). Without
# this, multi-line JSON output from `create-role`/`get-role` blocks the script
# at "-- More --" until the user presses a key.
$env:AWS_PAGER = ""

# -------- Profile resolution: env var > aws-account.yaml > fallback -------

function Get-ProfileFromYaml {
    param([string]$YamlPath)
    if (-not (Test-Path $YamlPath)) { return $null }
    $match = Select-String -Path $YamlPath -Pattern '^profile:\s*(\S+)' | Select-Object -First 1
    if ($match) { return $match.Matches[0].Groups[1].Value }
    return $null
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Profile_ = if ($env:AWS_PROFILE) {
    $env:AWS_PROFILE
} else {
    $fromYaml = Get-ProfileFromYaml (Join-Path $RepoRoot "aws-account.yaml")
    if ($fromYaml) { $fromYaml } else { "caravan-poc" }
}
$Region = if ($env:AWS_REGION) { $env:AWS_REGION } else { "us-east-1" }
$ProjectTag = "daccord"
$RoleName = "DaccordBedrockBatchService"

# Wrapper to silence stderr from native commands without tripping
# PowerShell 5.1's NativeCommandError trap (which converts the stderr
# stream into a terminating error under $ErrorActionPreference='Stop').
# `cmd /c '... 2>nul'` redirects stderr inside cmd before PowerShell sees it.
function Invoke-AwsQuiet {
    param([string]$Cmd)
    cmd /c "$Cmd 2>nul"
}

# Write UTF-8 WITHOUT BOM. Windows PowerShell 5.1's `Set-Content -Encoding utf8`
# emits a BOM that the AWS CLI's JSON parser rejects with
# "MalformedPolicyDocument: This policy contains invalid Json". Go through .NET
# directly with UTF8Encoding($false) — the `$false` arg means "no BOM".
function Write-Utf8NoBom {
    param([string]$Path, [string]$Content)
    [System.IO.File]::WriteAllText($Path, $Content, (New-Object System.Text.UTF8Encoding $false))
}

# Fail fast on non-zero aws CLI exit (native commands don't trip $ErrorActionPreference).
function Assert-LastAwsOk {
    param([string]$Op)
    if ($LASTEXITCODE -ne 0) {
        throw "aws CLI '$Op' failed with exit code $LASTEXITCODE"
    }
}

Write-Host ">> Resolving account ID via profile=$Profile_ ..."
$AccountId = aws sts get-caller-identity --profile $Profile_ --query Account --output text
$Bucket = "daccord-dev-$AccountId"
Write-Host "   account_id=$AccountId  bucket=$Bucket  region=$Region"

# -------- 1. S3 bucket ----------------------------------------------------

Write-Host ">> Creating S3 bucket s3://$Bucket (idempotent) ..."
Invoke-AwsQuiet "aws s3api head-bucket --bucket $Bucket --profile $Profile_"
if ($LASTEXITCODE -eq 0) {
    Write-Host "   bucket already exists - skipping create"
} else {
    # us-east-1 is special: do NOT pass --create-bucket-configuration.
    if ($Region -eq "us-east-1") {
        aws s3api create-bucket `
            --bucket $Bucket `
            --region $Region `
            --profile $Profile_
    } else {
        aws s3api create-bucket `
            --bucket $Bucket `
            --region $Region `
            --create-bucket-configuration "LocationConstraint=$Region" `
            --profile $Profile_
    }
    Write-Host "   bucket created"
}

Write-Host ">> Enabling versioning on s3://$Bucket ..."
aws s3api put-bucket-versioning `
    --bucket $Bucket `
    --versioning-configuration Status=Enabled `
    --profile $Profile_

Write-Host ">> Tagging bucket Project=$ProjectTag ..."
aws s3api put-bucket-tagging `
    --bucket $Bucket `
    --tagging "TagSet=[{Key=Project,Value=$ProjectTag}]" `
    --profile $Profile_

# -------- 2. Bedrock batch service role -----------------------------------

$TrustPolicy = @"
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "bedrock.amazonaws.com" },
      "Action": "sts:AssumeRole",
      "Condition": {
        "StringEquals": { "aws:SourceAccount": "$AccountId" },
        "ArnLike": {
          "aws:SourceArn": "arn:aws:bedrock:${Region}:${AccountId}:model-invocation-job/*"
        }
      }
    }
  ]
}
"@

$InlinePolicy = @"
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::${Bucket}/*"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::${Bucket}"
    }
  ]
}
"@

# Windows CLI mangles inline multi-line JSON with embedded quotes
# (PowerShell + aws CLI argument parsing strip and re-quote inconsistently).
# Write the JSON to temp files and pass `file://` to aws — the canonical
# workaround on Windows. Forward slashes in the path so aws's url parser
# accepts it.
$TrustFile = New-TemporaryFile
$InlineFile = New-TemporaryFile
try {
    Write-Utf8NoBom -Path $TrustFile.FullName -Content $TrustPolicy
    Write-Utf8NoBom -Path $InlineFile.FullName -Content $InlinePolicy
    $TrustUri = "file://" + ($TrustFile.FullName -replace '\\', '/')
    $InlineUri = "file://" + ($InlineFile.FullName -replace '\\', '/')

    Write-Host ">> Creating IAM role $RoleName (idempotent) ..."
    Invoke-AwsQuiet "aws iam get-role --role-name $RoleName --profile $Profile_"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   role already exists - updating trust policy"
        aws iam update-assume-role-policy `
            --role-name $RoleName `
            --policy-document $TrustUri `
            --profile $Profile_
        Assert-LastAwsOk "update-assume-role-policy"
    } else {
        aws iam create-role `
            --role-name $RoleName `
            --assume-role-policy-document $TrustUri `
            --description "Assumed by Bedrock to read/write D'accord S3 batch I/O" `
            --tags "Key=Project,Value=$ProjectTag" `
            --profile $Profile_
        Assert-LastAwsOk "create-role"
        Write-Host "   role created"
    }

    Write-Host ">> Attaching inline S3 policy to role ..."
    aws iam put-role-policy `
        --role-name $RoleName `
        --policy-name "DaccordS3BatchIO" `
        --policy-document $InlineUri `
        --profile $Profile_
    Assert-LastAwsOk "put-role-policy"
} finally {
    Remove-Item $TrustFile -ErrorAction SilentlyContinue
    Remove-Item $InlineFile -ErrorAction SilentlyContinue
}

$RoleArn = aws iam get-role --role-name $RoleName --profile $Profile_ --query 'Role.Arn' --output text

Write-Host ""
Write-Host "================ D'accord AWS setup complete ================"
Write-Host "  Profile     : $Profile_"
Write-Host "  Region      : $Region"
Write-Host "  S3 bucket   : s3://$Bucket (versioning ON, Project=$ProjectTag)"
Write-Host "  Service role: $RoleArn"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Run 'docker compose run --rm root uv run python scripts/check_aws_setup.py'"
Write-Host "     to verify + see which Bedrock models still need use-case forms."
Write-Host "  2. Open AWS Console -> Bedrock -> us-east-1 -> Model access:"
Write-Host "     submit the use-case form for any models reported as not granted."
Write-Host "     Typically: Claude Haiku 4.5 (auto-approved <5 min); Llama 4"
Write-Host "     Scout/Maverick and Nova 2 Lite are usually instant-grant."
Write-Host "  3. Re-run check_aws_setup.py until all 4 F9 models report access OK."
