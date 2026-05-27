"""Tier-7A Bedrock-batch client — submit + poll + download per (pair, model).

Layered on `daccord.aws.m2`:
  - `m2` supplies pure-constant resource conventions (region, bucket pattern,
    role ARN format) and per-framework-pair S3 path helpers.
  - This module wraps boto3 (impure I/O) and ferries data between the local
    on-disk staging directory, the daccord S3 bucket, and Bedrock's
    `create_model_invocation_job` / `get_model_invocation_job` APIs.

Each batch job submits ONE (framework_pair, model) prompt set as a single
JSONL file to S3. The Bedrock service reads the JSONL, runs each record
through the named model, and writes one output file per input plus a
manifest to the output prefix. The poll path downloads + parses outputs
back into `EnsembleCandidate` rows.

Per-model input format divergence: Bedrock batch records use each model's
**native** input format (NOT the unified Converse API — Converse batch is
limited as of 2026-05). The `build_modelInput()` dispatch covers the four
F9 ensemble seats:

  - Anthropic Claude: `{anthropic_version, max_tokens, system, messages}`
  - Meta Llama 4: native `messages` shape with `inferenceConfig`
  - Amazon Nova 2: Bedrock Converse-compatible `{schemaVersion, system,
    messages, inferenceConfig}` shape (Nova 2 supports Converse natively)

Per-model OUTPUT divergence is similar — `parse_model_output()` handles
each seat's response envelope. Parse failures (truncated JSON, missing
fields) become `EnsembleCandidate.parse_error` instead of raising, so a
single malformed record doesn't tank the whole batch.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from daccord.aws import m2
from daccord.ensemble import EnsembleCandidate
from daccord.validation import ValidatedModel, validated

log = logging.getLogger("daccord.aws.batch")


# ─────────────────────────────────────────────────────────────────────────────
# Filename safety — full Bedrock model IDs contain `:` and `.` which are
# either invalid on Windows or noisy in S3 keys. `model_slug()` produces a
# stable filename-safe form for `data/ensemble/raw/{pair}__{slug}.jsonl`.
# ─────────────────────────────────────────────────────────────────────────────


@validated
def model_slug(model_id: str) -> str:
    """Map a full Bedrock model ID to a Windows-safe, S3-clean slug.

    `meta.llama4-scout-17b-instruct-v1:0` → `meta-llama4-scout-17b-instruct-v1-0`.
    Substitutes `:` and `.` with `-`; collapses consecutive dashes.
    """
    s = re.sub(r"[:.]", "-", model_id)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


# ─────────────────────────────────────────────────────────────────────────────
# Per-model modelInput builders — Bedrock batch uses each model's native
# input schema. Each helper returns a dict ready to drop into a record's
# `modelInput` field.
# ─────────────────────────────────────────────────────────────────────────────

ANTHROPIC_VERSION = "bedrock-2023-05-31"
"""Pinned for Claude on Bedrock — required by `modelInput`. Schema version
is decoupled from the Claude model version (Haiku 4.5 still uses this)."""

NOVA_SCHEMA_VERSION = "messages-v1"
"""Nova 2's modelInput schema. Identical envelope to Bedrock Converse — same
`system`+`messages`+`inferenceConfig` shape."""


@validated
def _build_claude_input(system: str, user: str, max_tokens: int) -> dict[str, Any]:
    """Anthropic Messages API shape — required for Claude on Bedrock."""
    return {
        "anthropic_version": ANTHROPIC_VERSION,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }


@validated
def _build_llama_input(system: str, user: str, max_tokens: int) -> dict[str, Any]:
    """Meta Llama 4 on Bedrock — uses Converse-style messages + inferenceConfig.

    Llama 4 Scout/Maverick are Converse-native (verified via Bedrock docs
    cutoff 2026-05). Native prompt-string format (`<|begin_of_text|>` etc.)
    still works but Converse-style is preferred — boto3 normalizes.
    """
    return {
        "messages": [
            {"role": "user", "content": [{"text": user}]},
        ],
        "system": [{"text": system}],
        "inferenceConfig": {"maxTokens": max_tokens},
    }


@validated
def _build_nova_input(system: str, user: str, max_tokens: int) -> dict[str, Any]:
    """Amazon Nova 2 — schemaVersion-tagged Converse-compatible shape."""
    return {
        "schemaVersion": NOVA_SCHEMA_VERSION,
        "system": [{"text": system}],
        "messages": [
            {"role": "user", "content": [{"text": user}]},
        ],
        "inferenceConfig": {"maxTokens": max_tokens},
    }


@validated
def build_modelInput(model: str, system: str, user: str, max_tokens: int) -> dict[str, Any]:
    """Dispatch to the right per-model builder by Bedrock model ID prefix.

    Raises ValueError if `model` doesn't match any of the four F9 seats —
    intentionally narrow: this module's job is the F9 ensemble only.
    """
    if model.startswith("anthropic."):
        return _build_claude_input(system, user, max_tokens)
    if model.startswith("meta.llama"):
        return _build_llama_input(system, user, max_tokens)
    if model.startswith("amazon.nova"):
        return _build_nova_input(system, user, max_tokens)
    raise ValueError(f"unsupported Bedrock model id: {model!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-(pair, model) prompt staging — input JSONL assembly.
# ─────────────────────────────────────────────────────────────────────────────


class BatchPrompt(ValidatedModel):
    """One source-clause prompt to be sent to one Bedrock model in a batch.

    `record_id` is the JSONL `recordId` field — must be unique within a
    single batch (we use `source_id` directly, which is unique per pair).
    The `EnsembleCandidate` fields are kept here so `parse_model_output`
    can stitch source-side context back into the output row.
    """

    record_id: str
    source_id: str
    source_jurisdiction: str
    source_framework: str
    source_citation_id: str
    source_mechanism: str
    target_jurisdiction: str
    target_framework: str
    system: str
    user: str
    max_tokens: int


@validated
def build_batch_jsonl(prompts: list[BatchPrompt], model: str) -> str:
    """Assemble a multi-line JSONL string ready to upload as Bedrock batch input.

    Each line is `{"recordId": "...", "modelInput": {...}}` — Bedrock's
    documented batch-input format. Lines are ordered by `record_id` for
    deterministic re-runs.
    """
    sorted_prompts = sorted(prompts, key=lambda p: p.record_id)
    lines: list[str] = []
    for prompt in sorted_prompts:
        record = {
            "recordId": prompt.record_id,
            "modelInput": build_modelInput(
                model=model,
                system=prompt.system,
                user=prompt.user,
                max_tokens=prompt.max_tokens,
            ),
        }
        lines.append(json.dumps(record, ensure_ascii=False))
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# S3 helpers — upload input JSONL, list + download outputs.
# ─────────────────────────────────────────────────────────────────────────────


@validated
def _split_s3_uri(uri: str) -> tuple[str, str]:
    """`s3://bucket/key/path` → `(bucket, "key/path")`."""
    assert uri.startswith("s3://"), f"not an s3 URI: {uri!r}"
    rest = uri[len("s3://") :]
    bucket, _, key = rest.partition("/")
    return (bucket, key)


def upload_jsonl(session: Any, s3_uri: str, content: str) -> None:
    """PUT `content` (UTF-8 JSONL string) to `s3_uri`."""
    bucket, key = _split_s3_uri(s3_uri)
    s3 = session.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=content.encode("utf-8"))
    log.info("[s3] uploaded %d bytes to %s", len(content.encode("utf-8")), s3_uri)


def list_output_keys(session: Any, s3_uri_prefix: str) -> list[str]:
    """List S3 keys under `s3_uri_prefix` (directory-style URI ending in `/`)."""
    bucket, prefix = _split_s3_uri(s3_uri_prefix)
    s3 = session.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    out: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            out.append(obj["Key"])
    return sorted(out)


def download_to_string(session: Any, bucket: str, key: str) -> str:
    """Download an S3 object's body as a UTF-8 string."""
    s3 = session.client("s3")
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return body.decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Submit / poll wrappers — thin shims around create_model_invocation_job +
# get_model_invocation_job that route through `m2.*` helpers for all URIs.
# ─────────────────────────────────────────────────────────────────────────────


@validated
def s3_prefix_for(
    account_id: str, framework_pair: str, model: str, *, smoke: bool
) -> tuple[str, str]:
    """Return `(input_uri, output_uri)` for one (pair, model) batch.

    Smoke runs use `s3://{bucket}/ensemble/smoke/{in,out}/...` so smoke
    artifacts never collide with the real `ensemble/{in,out}/...` prefix.
    """
    slug = model_slug(model)
    bucket = m2.bucket_name(account_id)
    if smoke:
        in_uri = f"s3://{bucket}/ensemble/smoke/in/{framework_pair}__{slug}.jsonl"
        out_uri = f"s3://{bucket}/ensemble/smoke/out/{framework_pair}__{slug}/"
    else:
        in_uri = m2.s3_input_prefix(account_id, framework_pair, slug)
        out_uri = m2.s3_output_prefix(account_id, framework_pair, slug)
    return (in_uri, out_uri)


def submit_job(
    session: Any,
    account_id: str,
    framework_pair: str,
    model: str,
    jsonl_content: str,
    *,
    smoke: bool = False,
    job_name_prefix: str = "daccord",
) -> str:
    """Upload `jsonl_content` to S3 + start a Bedrock batch job.

    Returns the `jobArn` from `create_model_invocation_job`. Job name is
    `{job_name_prefix}-{framework_pair}-{slug}` truncated to Bedrock's 63-char
    limit. The S3 input/output URIs come from `m2.s3_{input,output}_prefix`.
    """
    in_uri, out_uri = s3_prefix_for(account_id, framework_pair, model, smoke=smoke)
    upload_jsonl(session, in_uri, jsonl_content)

    slug = model_slug(model)
    # Bedrock job names: alphanumerics + hyphens, ≤ 63 chars. `framework_pair`
    # already uses underscores → strip them, lowercase.
    raw_name = f"{job_name_prefix}-{framework_pair}-{slug}".replace("_", "-").lower()
    job_name = raw_name[:63].rstrip("-")

    bedrock = session.client("bedrock", region_name=m2.REGION)
    response = bedrock.create_model_invocation_job(
        jobName=job_name,
        modelId=model,
        roleArn=m2.role_arn(account_id),
        inputDataConfig={"s3InputDataConfig": {"s3Uri": in_uri}},
        outputDataConfig={"s3OutputDataConfig": {"s3Uri": out_uri}},
        tags=[{"key": "Project", "value": m2.PROJECT_TAG}],
    )
    job_arn = response["jobArn"]
    log.info("[bedrock] submitted job %s for %s/%s", job_arn, framework_pair, slug)
    return job_arn


def get_job_status(session: Any, job_arn: str) -> dict[str, Any]:
    """Fetch one batch job's current state.

    Returns the raw `get_model_invocation_job` response dict. Callers inspect
    `status` (one of `Submitted`, `InProgress`, `Completed`, `Failed`,
    `Stopping`, `Stopped`, `PartiallyCompleted`, `Expired`, `Validating`,
    `Scheduled`).
    """
    bedrock = session.client("bedrock", region_name=m2.REGION)
    return bedrock.get_model_invocation_job(jobIdentifier=job_arn)


# ─────────────────────────────────────────────────────────────────────────────
# Per-model output parsers — each seat's response envelope differs.
# ─────────────────────────────────────────────────────────────────────────────


_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _extract_text_from_output(model: str, model_output: dict[str, Any]) -> str | None:
    """Pull the model's textual response out of its `modelOutput` envelope.

    Returns None when the structure doesn't match any known shape (rare —
    indicates a Bedrock API drift; caller logs + records as parse_error).
    """
    if model.startswith("anthropic."):
        # Claude on Bedrock: {"content": [{"type": "text", "text": "..."}], ...}
        content = model_output.get("content")
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and "text" in first:
                return str(first["text"])
        return None
    if model.startswith("meta.llama"):
        # Llama 4 via Converse-style output:
        # {"output": {"message": {"content": [{"text": "..."}]}}, ...}
        output_block = model_output.get("output")
        if isinstance(output_block, dict):
            message = output_block.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, list) and content:
                    first = content[0]
                    if isinstance(first, dict) and "text" in first:
                        return str(first["text"])
        # Fallback to native Llama format if Converse style wasn't returned.
        if "generation" in model_output:
            return str(model_output["generation"])
        return None
    if model.startswith("amazon.nova"):
        # Nova 2: same Converse-style envelope as Llama.
        output_block = model_output.get("output")
        if isinstance(output_block, dict):
            message = output_block.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, list) and content:
                    first = content[0]
                    if isinstance(first, dict) and "text" in first:
                        return str(first["text"])
        return None
    return None


def _parse_candidate_json(text: str) -> tuple[dict[str, str], str | None]:
    """Parse the model's text into the `{citation_id, target_mechanism,
    mapping_justification}` triple.

    Models may wrap JSON in code fences, prose, or partial markdown. We
    extract the first top-level `{...}` block and json-loads it. Returns
    `(fields, parse_error)` — `fields` has empty-string defaults when a key
    is missing; `parse_error` is non-None when JSON couldn't be recovered.
    """
    candidate_fields = {"citation_id": "", "target_mechanism": "", "mapping_justification": ""}
    # Try the whole text first — JSON-only responses are most common from the
    # ensemble prompt's strict schema instruction.
    for blob in (text, *(_JSON_OBJECT_RE.findall(text) or [])):
        try:
            parsed = json.loads(blob)
            if isinstance(parsed, dict):
                for key in candidate_fields:
                    if key in parsed and parsed[key] is not None:
                        candidate_fields[key] = str(parsed[key])
                return (candidate_fields, None)
        except json.JSONDecodeError:
            continue
    return (candidate_fields, f"could not parse JSON from model output: {text[:120]!r}")


def parse_model_output(
    *,
    record_id: str,
    raw_output: dict[str, Any],
    prompts_by_record_id: dict[str, BatchPrompt],
    model: str,
) -> EnsembleCandidate | None:
    """Turn one Bedrock output record into an `EnsembleCandidate`.

    `raw_output` is one full record from a Bedrock batch output JSONL: it has
    a `recordId` (matched against `prompts_by_record_id`) and a `modelOutput`
    block whose shape depends on the model.

    Returns None when the record's `recordId` isn't in the prompt set (Bedrock
    occasionally emits manifest rows that aren't per-record outputs — caller
    skips them).
    """
    prompt = prompts_by_record_id.get(record_id)
    if prompt is None:
        return None

    model_output = raw_output.get("modelOutput")
    if not isinstance(model_output, dict):
        return EnsembleCandidate(
            source_id=prompt.source_id,
            source_jurisdiction=prompt.source_jurisdiction,
            source_framework=prompt.source_framework,
            source_citation_id=prompt.source_citation_id,
            source_mechanism=prompt.source_mechanism,
            target_jurisdiction=prompt.target_jurisdiction,
            target_framework=prompt.target_framework,
            model=model,
            citation_id="",
            target_mechanism="",
            mapping_justification="",
            parse_error="missing modelOutput in Bedrock response",
        )

    text = _extract_text_from_output(model, model_output)
    if text is None:
        return EnsembleCandidate(
            source_id=prompt.source_id,
            source_jurisdiction=prompt.source_jurisdiction,
            source_framework=prompt.source_framework,
            source_citation_id=prompt.source_citation_id,
            source_mechanism=prompt.source_mechanism,
            target_jurisdiction=prompt.target_jurisdiction,
            target_framework=prompt.target_framework,
            model=model,
            citation_id="",
            target_mechanism="",
            mapping_justification="",
            parse_error=f"unrecognised modelOutput shape for {model}",
        )

    fields, parse_error = _parse_candidate_json(text)
    return EnsembleCandidate(
        source_id=prompt.source_id,
        source_jurisdiction=prompt.source_jurisdiction,
        source_framework=prompt.source_framework,
        source_citation_id=prompt.source_citation_id,
        source_mechanism=prompt.source_mechanism,
        target_jurisdiction=prompt.target_jurisdiction,
        target_framework=prompt.target_framework,
        model=model,
        citation_id=fields["citation_id"],
        target_mechanism=fields["target_mechanism"],
        mapping_justification=fields["mapping_justification"],
        parse_error=parse_error,
    )


def download_and_parse_outputs(
    session: Any,
    output_s3_uri: str,
    prompts: list[BatchPrompt],
    model: str,
) -> tuple[list[EnsembleCandidate], dict[str, int]]:
    """Pull all output records from `output_s3_uri` + parse to candidates.

    Bedrock writes one output file per input file (mirroring the JSONL
    structure), plus a manifest. We list the prefix, filter to `.jsonl.out`
    files (the per-record outputs), download each, parse each line, and
    map back to `EnsembleCandidate` via `recordId`.

    Returns `(candidates, stats)` where stats has
    `{"total": N, "parsed_ok": K, "parse_errors": E, "missing": M}` —
    counts useful for the orchestrator's per-batch log line.
    """
    prompts_by_id = {p.record_id: p for p in prompts}
    keys = list_output_keys(session, output_s3_uri)
    output_keys = [k for k in keys if k.endswith(".jsonl.out")]
    bucket, _ = _split_s3_uri(output_s3_uri)

    candidates: list[EnsembleCandidate] = []
    parse_errors = 0
    seen_record_ids: set[str] = set()

    for key in output_keys:
        body = download_to_string(session, bucket, key)
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            record_id = raw.get("recordId")
            if not isinstance(record_id, str):
                continue
            seen_record_ids.add(record_id)
            cand = parse_model_output(
                record_id=record_id,
                raw_output=raw,
                prompts_by_record_id=prompts_by_id,
                model=model,
            )
            if cand is not None:
                candidates.append(cand)
                if cand.parse_error is not None:
                    parse_errors += 1

    missing_record_ids = set(prompts_by_id.keys()) - seen_record_ids
    stats = {
        "total": len(prompts),
        "parsed_ok": len([c for c in candidates if c.parse_error is None]),
        "parse_errors": parse_errors,
        "missing": len(missing_record_ids),
    }
    return (candidates, stats)


# ─────────────────────────────────────────────────────────────────────────────
# Synchronous InvokeModel path — fallback for accounts without Bedrock-batch
# enablement. Pricing is 2× the batch tier (no batch discount). Used by
# `scripts/run_ensemble.py run-sync`. Same `parse_model_output` recovers
# `EnsembleCandidate` from the response.
# ─────────────────────────────────────────────────────────────────────────────


_RETRYABLE_INVOKE_ERRORS = (
    "ThrottlingException",
    "ModelTimeoutException",
    "ServiceUnavailableException",
    "InternalServerException",
)


@validated
def inference_profile_id(model_id: str) -> str:
    """Map a bare foundation-model ID to its US-geo CRIS inference profile ID.

    On-demand `bedrock-runtime:InvokeModel` rejects the bare F9 model IDs with
    a `ValidationException` ("Invocation … with on-demand throughput isn't
    supported") — these models route through Cross-Region Inference (CRIS)
    profiles. The `us.` prefix is AWS's stable convention for the US-geo
    profile, verified via `bedrock:ListInferenceProfiles` (see
    `scripts/check_aws_setup.py`).

    The bare model_id remains canonical everywhere else (costs config keys,
    `EnsembleCandidate.model`, m2.F9_BEDROCK_MODELS); this helper is only
    consulted at the InvokeModel boundary."""
    return f"us.{model_id}"


def invoke_sync(
    session: Any,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    *,
    max_retries: int = 5,
    base_backoff_s: float = 2.0,
) -> dict[str, Any]:
    """Call `bedrock-runtime:InvokeModel` synchronously, with exponential backoff.

    Returns the parsed response body (the same `modelOutput` shape we'd see in
    batch output's per-record `modelOutput` field) — so `parse_model_output`
    can consume the result identically across sync + batch paths.

    Retries on transient errors (`ThrottlingException`, model timeouts,
    `ServiceUnavailable`, `InternalServer`). 5 retries with 2s base backoff →
    max wait ≈ 2+4+8+16+32 = 62s before giving up; the final exception
    propagates so the orchestrator can log and continue.
    """
    body = json.dumps(build_modelInput(model, system, user, max_tokens))
    runtime = session.client("bedrock-runtime", region_name=m2.REGION)
    # F9 models require the CRIS inference-profile ID for on-demand invokes.
    invocation_id = inference_profile_id(model)

    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = runtime.invoke_model(
                modelId=invocation_id, body=body, contentType="application/json"
            )
            raw = resp["body"].read()
            return json.loads(raw)
        except Exception as exc:
            err_code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if err_code in _RETRYABLE_INVOKE_ERRORS and attempt < max_retries:
                wait = base_backoff_s * (2**attempt)
                log.warning(
                    "[invoke_sync] %s on attempt %d/%d; sleeping %.1fs",
                    err_code,
                    attempt + 1,
                    max_retries + 1,
                    wait,
                )
                time.sleep(wait)
                last_exc = exc
                continue
            raise
    # Defensive: should be unreachable due to the `raise` in the else branch.
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("invoke_sync: retry loop exhausted without exception")


@validated
def candidate_from_invoke_response(
    *,
    prompt: BatchPrompt,
    response_body: dict[str, Any],
    model: str,
) -> EnsembleCandidate:
    """Adapter: wrap a sync InvokeModel response in the batch parser shape.

    The batch path's `parse_model_output` expects a record like
    `{"recordId": ..., "modelOutput": {...}}`; we synthesize that envelope
    around the sync response body. This is the single point of convergence
    between sync + batch — every other downstream step (write_candidates_jsonl,
    cost recording, tier_ensemble.py consumption) is identical.
    """
    cand = parse_model_output(
        record_id=prompt.record_id,
        raw_output={"recordId": prompt.record_id, "modelOutput": response_body},
        prompts_by_record_id={prompt.record_id: prompt},
        model=model,
    )
    if cand is None:
        # Shouldn't happen: parse_model_output only returns None when the
        # record_id isn't in the prompts map, which we just constructed.
        raise RuntimeError(f"unexpected None candidate for record_id={prompt.record_id}")
    return cand


@validated
def write_candidates_jsonl(path: Path, candidates: list[EnsembleCandidate]) -> None:
    """Atomic-write `EnsembleCandidate` rows to `path`, sorted by `source_id`.

    Same temp-file + replace pattern as `scripts/tier_ensemble.py:_write_tiered`
    so re-runs produce byte-identical output.
    """
    sorted_rows = sorted(candidates, key=lambda c: c.source_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in sorted_rows:
            f.write(row.model_dump_json())
            f.write("\n")
    tmp.replace(path)
