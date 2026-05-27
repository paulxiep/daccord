"""Tier-7A Bedrock-batch client tests.

Covers:
  - Per-model `modelInput` builders return the right native shape per F9 seat.
  - `build_batch_jsonl` produces deterministic, line-sorted JSONL.
  - Submit-path calls boto3 with the right URIs (mocked Bedrock + S3 client).
  - Output parsing recovers candidates from each model's response envelope,
    sets `parse_error` on malformed JSON instead of crashing.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from daccord.aws import batch
from daccord.aws.batch import (
    BatchPrompt,
    build_batch_jsonl,
    build_modelInput,
    download_and_parse_outputs,
    model_slug,
    parse_model_output,
    s3_prefix_for,
    submit_job,
)


def _make_prompt(record_id: str = "src_1", user: str = "user text") -> BatchPrompt:
    return BatchPrompt(
        record_id=record_id,
        source_id=record_id,
        source_jurisdiction="eu",
        source_framework="gdpr",
        source_citation_id="6",
        source_mechanism="Lawful processing.",
        target_jurisdiction="sg",
        target_framework="pdpa_sg",
        system="System prompt.",
        user=user,
        max_tokens=200,
    )


class TestBuildModelInput:
    def test_claude_input_shape(self) -> None:
        out = build_modelInput(
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            system="sys",
            user="usr",
            max_tokens=100,
        )
        assert out["anthropic_version"] == batch.ANTHROPIC_VERSION
        assert out["max_tokens"] == 100
        assert out["system"] == "sys"
        assert out["messages"] == [{"role": "user", "content": "usr"}]

    def test_llama_input_shape_converse_style(self) -> None:
        out = build_modelInput(
            model="meta.llama4-scout-17b-instruct-v1:0",
            system="sys",
            user="usr",
            max_tokens=100,
        )
        assert out["system"] == [{"text": "sys"}]
        assert out["messages"] == [{"role": "user", "content": [{"text": "usr"}]}]
        assert out["inferenceConfig"] == {"maxTokens": 100}

    def test_nova_input_shape(self) -> None:
        out = build_modelInput(
            model="amazon.nova-2-lite-v1:0",
            system="sys",
            user="usr",
            max_tokens=100,
        )
        assert out["schemaVersion"] == batch.NOVA_SCHEMA_VERSION
        assert out["system"] == [{"text": "sys"}]
        assert out["messages"] == [{"role": "user", "content": [{"text": "usr"}]}]
        assert out["inferenceConfig"] == {"maxTokens": 100}

    def test_unsupported_model_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported"):
            build_modelInput(model="cohere.command-r-plus", system="", user="", max_tokens=10)


class TestBuildBatchJsonl:
    def test_record_lines_have_unique_ids(self) -> None:
        prompts = [_make_prompt("src_2"), _make_prompt("src_1")]
        content = build_batch_jsonl(prompts, model="anthropic.claude-haiku-4-5-20251001-v1:0")
        lines = [json.loads(ln) for ln in content.splitlines() if ln]
        # Sorted by record_id for byte-identical reruns.
        assert [ln["recordId"] for ln in lines] == ["src_1", "src_2"]

    def test_modelInput_dispatches_per_model(self) -> None:
        prompts = [_make_prompt("src_1")]
        nova = build_batch_jsonl(prompts, model="amazon.nova-2-lite-v1:0")
        nova_line = json.loads(nova.splitlines()[0])
        assert "schemaVersion" in nova_line["modelInput"]

        claude = build_batch_jsonl(prompts, model="anthropic.claude-haiku-4-5-20251001-v1:0")
        claude_line = json.loads(claude.splitlines()[0])
        assert "anthropic_version" in claude_line["modelInput"]


class TestS3PrefixFor:
    def test_real_run_uses_m2_helpers(self) -> None:
        in_uri, out_uri = s3_prefix_for(
            "123456789012",
            "gdpr__pdpa_sg",
            "meta.llama4-scout-17b-instruct-v1:0",
            smoke=False,
        )
        assert in_uri == (
            "s3://daccord-dev-123456789012/ensemble/in/"
            "gdpr__pdpa_sg__meta-llama4-scout-17b-instruct-v1-0.jsonl"
        )
        assert out_uri == (
            "s3://daccord-dev-123456789012/ensemble/out/"
            "gdpr__pdpa_sg__meta-llama4-scout-17b-instruct-v1-0/"
        )

    def test_smoke_uses_smoke_prefix(self) -> None:
        in_uri, out_uri = s3_prefix_for(
            "123456789012",
            "gdpr__pdpa_sg",
            "amazon.nova-2-lite-v1:0",
            smoke=True,
        )
        assert "/ensemble/smoke/in/" in in_uri
        assert "/ensemble/smoke/out/" in out_uri
        # Smoke output never overlaps with real output prefix.
        assert "/ensemble/out/" not in out_uri


class TestSubmitJob:
    def test_uploads_jsonl_and_calls_bedrock(self) -> None:
        session = MagicMock()
        s3_client = MagicMock()
        bedrock_client = MagicMock()
        bedrock_client.create_model_invocation_job.return_value = {
            "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/abc123"
        }
        session.client.side_effect = lambda service, region_name=None: (
            s3_client if service == "s3" else bedrock_client
        )

        jsonl = '{"recordId":"src_1","modelInput":{"x":1}}\n'
        job_arn = submit_job(
            session=session,
            account_id="123456789012",
            framework_pair="gdpr__pdpa_sg",
            model="meta.llama4-scout-17b-instruct-v1:0",
            jsonl_content=jsonl,
            smoke=False,
        )

        # S3 upload: bucket + key derived from m2.bucket_name + slug.
        s3_client.put_object.assert_called_once()
        put_kwargs = s3_client.put_object.call_args.kwargs
        assert put_kwargs["Bucket"] == "daccord-dev-123456789012"
        assert "gdpr__pdpa_sg__meta-llama4-scout-17b-instruct-v1-0.jsonl" in put_kwargs["Key"]
        assert put_kwargs["Body"] == jsonl.encode("utf-8")

        # Bedrock job: roleArn + URIs come from m2.* helpers.
        bedrock_client.create_model_invocation_job.assert_called_once()
        bedrock_kwargs = bedrock_client.create_model_invocation_job.call_args.kwargs
        assert (
            bedrock_kwargs["roleArn"] == "arn:aws:iam::123456789012:role/DaccordBedrockBatchService"
        )
        assert bedrock_kwargs["modelId"] == "meta.llama4-scout-17b-instruct-v1:0"
        in_uri = bedrock_kwargs["inputDataConfig"]["s3InputDataConfig"]["s3Uri"]
        out_uri = bedrock_kwargs["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
        assert in_uri.startswith("s3://daccord-dev-123456789012/ensemble/in/")
        assert out_uri.endswith("/")
        assert bedrock_kwargs["tags"] == [{"key": "Project", "value": "daccord"}]
        assert job_arn.endswith("abc123")


class TestParseModelOutput:
    def _prompts_map(self) -> dict[str, BatchPrompt]:
        return {"src_1": _make_prompt("src_1")}

    def test_claude_response(self) -> None:
        raw: dict[str, Any] = {
            "recordId": "src_1",
            "modelOutput": {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            '{"citation_id": "13",'
                            ' "target_mechanism": "Consent is required",'
                            ' "mapping_justification": "Maps GDPR Art 6 to PDPA-SG s.13"}'
                        ),
                    }
                ]
            },
        }
        cand = parse_model_output(
            record_id="src_1",
            raw_output=raw,
            prompts_by_record_id=self._prompts_map(),
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
        )
        assert cand is not None
        assert cand.citation_id == "13"
        assert "Consent" in cand.target_mechanism
        assert cand.parse_error is None

    def test_llama_response_converse_style(self) -> None:
        raw = {
            "recordId": "src_1",
            "modelOutput": {
                "output": {
                    "message": {
                        "content": [
                            {
                                "text": (
                                    '{"citation_id": "13",'
                                    ' "target_mechanism": "Consent.",'
                                    ' "mapping_justification": "Maps."}'
                                )
                            }
                        ]
                    }
                }
            },
        }
        cand = parse_model_output(
            record_id="src_1",
            raw_output=raw,
            prompts_by_record_id=self._prompts_map(),
            model="meta.llama4-scout-17b-instruct-v1:0",
        )
        assert cand is not None
        assert cand.citation_id == "13"
        assert cand.parse_error is None

    def test_nova_response_converse_style(self) -> None:
        raw = {
            "recordId": "src_1",
            "modelOutput": {
                "output": {
                    "message": {
                        "content": [
                            {
                                "text": (
                                    '{"citation_id": "13",'
                                    ' "target_mechanism": "Consent.",'
                                    ' "mapping_justification": "Maps."}'
                                )
                            }
                        ]
                    }
                }
            },
        }
        cand = parse_model_output(
            record_id="src_1",
            raw_output=raw,
            prompts_by_record_id=self._prompts_map(),
            model="amazon.nova-2-lite-v1:0",
        )
        assert cand is not None
        assert cand.citation_id == "13"
        assert cand.parse_error is None

    def test_malformed_json_records_parse_error(self) -> None:
        raw = {
            "recordId": "src_1",
            "modelOutput": {"content": [{"text": "this is not JSON at all"}]},
        }
        cand = parse_model_output(
            record_id="src_1",
            raw_output=raw,
            prompts_by_record_id=self._prompts_map(),
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
        )
        assert cand is not None
        assert cand.parse_error is not None
        assert cand.citation_id == ""

    def test_missing_modelOutput_records_parse_error(self) -> None:
        cand = parse_model_output(
            record_id="src_1",
            raw_output={"recordId": "src_1"},  # no modelOutput
            prompts_by_record_id=self._prompts_map(),
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
        )
        assert cand is not None
        assert cand.parse_error is not None and "missing modelOutput" in cand.parse_error

    def test_unknown_record_id_returns_none(self) -> None:
        cand = parse_model_output(
            record_id="not-in-prompts",
            raw_output={"recordId": "not-in-prompts", "modelOutput": {}},
            prompts_by_record_id=self._prompts_map(),
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
        )
        assert cand is None

    def test_claude_response_with_code_fence_wrapper(self) -> None:
        """Models occasionally wrap JSON in ```json fences — extractor handles it."""
        raw = {
            "recordId": "src_1",
            "modelOutput": {
                "content": [
                    {
                        "text": (
                            "Here is my answer:\n```json\n"
                            '{"citation_id": "13", "target_mechanism": "Consent.",'
                            ' "mapping_justification": "Maps."}\n```'
                        )
                    }
                ]
            },
        }
        cand = parse_model_output(
            record_id="src_1",
            raw_output=raw,
            prompts_by_record_id=self._prompts_map(),
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
        )
        assert cand is not None
        assert cand.citation_id == "13"
        assert cand.parse_error is None


class TestDownloadAndParseOutputs:
    def test_round_trips_one_output_file(self) -> None:
        session = MagicMock()
        s3_client = MagicMock()
        session.client.return_value = s3_client

        # list_objects_v2 paginator → one .jsonl.out file
        paginator = MagicMock()
        out_prefix = "ensemble/smoke/out/gdpr__pdpa_sg__meta-llama4-scout-17b-instruct-v1-0/"
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": f"{out_prefix}manifest.json.out"},
                    {"Key": f"{out_prefix}in.jsonl.out"},
                ]
            }
        ]
        s3_client.get_paginator.return_value = paginator

        # download_to_string returns a fake JSONL with one record.
        body_text = (
            json.dumps(
                {
                    "recordId": "src_1",
                    "modelOutput": {
                        "output": {
                            "message": {
                                "content": [
                                    {
                                        "text": (
                                            '{"citation_id":"13",'
                                            '"target_mechanism":"Consent.",'
                                            '"mapping_justification":"Maps."}'
                                        )
                                    }
                                ]
                            }
                        }
                    },
                }
            )
            + "\n"
        )
        s3_client.get_object.return_value = {
            "Body": MagicMock(read=lambda: body_text.encode("utf-8"))
        }

        candidates, stats = download_and_parse_outputs(
            session=session,
            output_s3_uri=f"s3://daccord-dev-123456789012/{out_prefix}",
            prompts=[_make_prompt("src_1")],
            model="meta.llama4-scout-17b-instruct-v1:0",
        )

        assert stats == {"total": 1, "parsed_ok": 1, "parse_errors": 0, "missing": 0}
        assert len(candidates) == 1
        assert candidates[0].citation_id == "13"

    def test_missing_record_shows_in_stats(self) -> None:
        session = MagicMock()
        s3_client = MagicMock()
        session.client.return_value = s3_client
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Contents": []}]
        s3_client.get_paginator.return_value = paginator

        _, stats = download_and_parse_outputs(
            session=session,
            output_s3_uri="s3://daccord-dev-123456789012/ensemble/smoke/out/foo/",
            prompts=[_make_prompt("src_1"), _make_prompt("src_2")],
            model="amazon.nova-2-lite-v1:0",
        )
        assert stats["missing"] == 2
        assert stats["parsed_ok"] == 0


class TestInvokeSync:
    def test_calls_bedrock_runtime_with_native_input(self) -> None:
        session = MagicMock()
        runtime = MagicMock()
        session.client.return_value = runtime
        body_text = json.dumps(
            {
                "content": [
                    {
                        "text": (
                            '{"citation_id":"13","target_mechanism":"x",'
                            '"mapping_justification":"y"}'
                        )
                    }
                ]
            }
        )
        runtime.invoke_model.return_value = {
            "body": MagicMock(read=lambda: body_text.encode("utf-8"))
        }

        out = batch.invoke_sync(
            session=session,
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            system="sys",
            user="usr",
            max_tokens=100,
        )

        runtime.invoke_model.assert_called_once()
        kwargs = runtime.invoke_model.call_args.kwargs
        # F9 models require the `us.` CRIS profile prefix for on-demand invokes.
        assert kwargs["modelId"] == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        assert kwargs["contentType"] == "application/json"
        # Body is Claude's native input shape — confirms build_modelInput dispatch.
        sent = json.loads(kwargs["body"])
        assert sent["anthropic_version"] == batch.ANTHROPIC_VERSION
        assert sent["max_tokens"] == 100
        assert out["content"][0]["text"].startswith("{")

    def test_retries_on_throttling(self) -> None:
        """ThrottlingException retries with backoff; success after retry returns body."""
        session = MagicMock()
        runtime = MagicMock()
        session.client.return_value = runtime
        # First call raises Throttling, second succeeds.
        throttle_exc = Exception()
        throttle_exc.response = {"Error": {"Code": "ThrottlingException"}}  # type: ignore[attr-defined]
        body_text = json.dumps({"content": [{"text": "ok"}]})
        runtime.invoke_model.side_effect = [
            throttle_exc,
            {"body": MagicMock(read=lambda: body_text.encode("utf-8"))},
        ]

        # Patch time.sleep so the test is fast.
        with patch.object(batch.time, "sleep", return_value=None):
            out = batch.invoke_sync(
                session=session,
                model="amazon.nova-2-lite-v1:0",
                system="sys",
                user="usr",
                max_tokens=50,
                max_retries=2,
                base_backoff_s=0.01,
            )
        assert runtime.invoke_model.call_count == 2
        assert out["content"][0]["text"] == "ok"

    def test_non_retryable_error_propagates(self) -> None:
        session = MagicMock()
        runtime = MagicMock()
        session.client.return_value = runtime
        access_denied = Exception("AccessDenied")
        access_denied.response = {"Error": {"Code": "AccessDeniedException"}}  # type: ignore[attr-defined]
        runtime.invoke_model.side_effect = access_denied

        with pytest.raises(Exception, match="AccessDenied"):
            batch.invoke_sync(
                session=session,
                model="amazon.nova-2-lite-v1:0",
                system="sys",
                user="usr",
                max_tokens=50,
                max_retries=2,
                base_backoff_s=0.01,
            )
        # No retries on non-retryable errors.
        assert runtime.invoke_model.call_count == 1


class TestCandidateFromInvokeResponse:
    def test_round_trip(self) -> None:
        prompt = _make_prompt("src_1")
        response_body = {
            "content": [
                {
                    "text": (
                        '{"citation_id":"13","target_mechanism":"Consent.",'
                        '"mapping_justification":"Maps."}'
                    )
                }
            ]
        }
        cand = batch.candidate_from_invoke_response(
            prompt=prompt,
            response_body=response_body,
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
        )
        assert cand.citation_id == "13"
        assert cand.source_id == "src_1"
        assert cand.parse_error is None


class TestModelSlugConsistency:
    def test_slug_matches_aws_m2_test_expectations(self) -> None:
        # If this regresses, anything written to disk + S3 under the prior slug
        # becomes orphaned. Same guarantee as tests/test_aws_m2.py — we re-pin
        # at the batch module's import site too.
        assert (
            model_slug("meta.llama4-scout-17b-instruct-v1:0")
            == "meta-llama4-scout-17b-instruct-v1-0"
        )
        assert model_slug("amazon.nova-2-lite-v1:0") == "amazon-nova-2-lite-v1-0"
