# Tier-6C teardown for D'accord M2 AWS resources.
#
# PowerShell port of aws_teardown.sh for Windows users.
#
# Removes:
#   1. S3 bucket `daccord-dev-{account_id}` AND all objects (including
#      versioned objects) inside it. Destructive.
#   2. IAM role `DaccordBedrockBatchService` + its inline policy.
#
# Profile resolution: same as aws_setup.ps1 (env > aws-account.yaml > fallback).
#
# Usage:
#   .\scripts\aws_teardown.ps1

$ErrorActionPreference = "Stop"

# Disable AWS CLI's interactive pager.
$env:AWS_PAGER = ""

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
$RoleName = "DaccordBedrockBatchService"

function Invoke-AwsQuiet {
    param([string]$Cmd)
    cmd /c "$Cmd 2>nul"
}

# Windows PowerShell 5.1's `Set-Content -Encoding utf8` adds a BOM that
# AWS CLI's JSON parser rejects. Use .NET UTF8Encoding($false) instead.
function Write-Utf8NoBom {
    param([string]$Path, [string]$Content)
    [System.IO.File]::WriteAllText($Path, $Content, (New-Object System.Text.UTF8Encoding $false))
}

$AccountId = aws sts get-caller-identity --profile $Profile_ --query Account --output text
$Bucket = "daccord-dev-$AccountId"

Write-Host ">> Tearing down D'accord AWS resources in account $AccountId ..."

# -------- 1. S3 bucket (delete all versions + bucket) ---------------------

Invoke-AwsQuiet "aws s3api head-bucket --bucket $Bucket --profile $Profile_"
if ($LASTEXITCODE -eq 0) {
    Write-Host ">> Deleting all object versions in s3://$Bucket ..."
    $Versions = aws s3api list-object-versions `
        --bucket $Bucket `
        --output json `
        --profile $Profile_ `
        --query '{Objects: Versions[].{Key: Key, VersionId: VersionId}}' `
        | ConvertFrom-Json
    if ($Versions.Objects) {
        $Payload = $Versions | ConvertTo-Json -Compress -Depth 10
        $TmpFile = New-TemporaryFile
        Write-Utf8NoBom -Path $TmpFile.FullName -Content $Payload
        aws s3api delete-objects `
            --bucket $Bucket `
            --delete "file://$TmpFile" `
            --profile $Profile_ | Out-Null
        Remove-Item $TmpFile
    }

    $Markers = aws s3api list-object-versions `
        --bucket $Bucket `
        --output json `
        --profile $Profile_ `
        --query '{Objects: DeleteMarkers[].{Key: Key, VersionId: VersionId}}' `
        | ConvertFrom-Json
    if ($Markers.Objects) {
        $Payload = $Markers | ConvertTo-Json -Compress -Depth 10
        $TmpFile = New-TemporaryFile
        Write-Utf8NoBom -Path $TmpFile.FullName -Content $Payload
        aws s3api delete-objects `
            --bucket $Bucket `
            --delete "file://$TmpFile" `
            --profile $Profile_ | Out-Null
        Remove-Item $TmpFile
    }

    Write-Host ">> Deleting bucket s3://$Bucket ..."
    aws s3api delete-bucket --bucket $Bucket --region $Region --profile $Profile_
} else {
    Write-Host "   bucket s3://$Bucket not found - skipping"
}

# -------- 2. IAM role -----------------------------------------------------

Invoke-AwsQuiet "aws iam get-role --role-name $RoleName --profile $Profile_"
if ($LASTEXITCODE -eq 0) {
    Write-Host ">> Removing inline policy from role $RoleName ..."
    Invoke-AwsQuiet "aws iam delete-role-policy --role-name $RoleName --policy-name DaccordS3BatchIO --profile $Profile_"

    Write-Host ">> Deleting IAM role $RoleName ..."
    aws iam delete-role --role-name $RoleName --profile $Profile_
} else {
    Write-Host "   role $RoleName not found - skipping"
}

Write-Host ""
Write-Host "================ D'accord AWS teardown complete ================"
Write-Host "  Bucket s3://$Bucket removed (if existed)"
Write-Host "  Role $RoleName removed (if existed)"
Write-Host "  Caravan-poc profile + budget untouched"
