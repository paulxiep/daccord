"""Regression guards for the M2 AWS-resource convention module.

These constants + helpers are the contract that ties tier 6C verification,
tier 7A Bedrock-batch submit/poll, and any future M2 tooling together —
changing a constant here must be a deliberate edit, not a typo.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

from daccord.aws import m2


class TestConstants:
    def test_region_is_us_east_1(self) -> None:
        # us-east-1 is the only region with Llama 4 access as of 2026-05.
        # Changing this requires re-verifying Llama 4 regional availability.
        assert m2.REGION == "us-east-1"

    def test_project_tag_lowercase(self) -> None:
        # AWS cost-allocation tag matches the convention in
        # docs/development_plan.md §6 + the existing $50/month budget filter.
        assert m2.PROJECT_TAG == "daccord"

    def test_role_name_camelcase(self) -> None:
        # IAM role name committed in scripts/aws_setup.sh/.ps1 and live in
        # the user's AWS account — must match those scripts byte-for-byte.
        assert m2.ROLE_NAME == "DaccordBedrockBatchService"

    def test_inline_policy_name(self) -> None:
        assert m2.INLINE_POLICY_NAME == "DaccordS3BatchIO"

    def test_f9_ensemble_has_four_seats(self) -> None:
        # F9-E ensemble is 4 seats by design. Tier 8 HIGH=4/4 agreement
        # depends on this length. If you drop to 3, tier 8 needs retiering.
        assert len(m2.F9_BEDROCK_MODELS) == 4

    def test_f9_ensemble_member_families(self) -> None:
        # Sanity: the 4 seats span 3 families (2 Meta MoEs + Anthropic + Amazon).
        # Catches accidental drift if someone replaces Maverick with another Llama.
        prefixes = {model.split(".")[0] for model in m2.F9_BEDROCK_MODELS}
        assert prefixes == {"meta", "anthropic", "amazon"}

    def test_f9_ensemble_contains_llama_4_scout(self) -> None:
        assert "meta.llama4-scout-17b-instruct-v1:0" in m2.F9_BEDROCK_MODELS

    def test_f9_pricing_keys_match_costs_config(self) -> None:
        # Each F9 model must have a pricing entry in costs/config.toml under
        # [pricing.bedrock_batch.*] — otherwise preflight() will raise
        # UnknownModel at tier 7A submit time.
        from daccord.costs.config import load_config

        cfg = load_config()
        priced = set(cfg.pricing.get("bedrock_batch", {}).keys())
        for model in m2.F9_BEDROCK_MODELS:
            assert model in priced, f"missing pricing for {model} in costs/config.toml"


class TestBucketArnHelpers:
    def test_bucket_name_pattern(self) -> None:
        assert m2.bucket_name("123456789012") == "daccord-dev-123456789012"

    def test_bucket_arn_derives_from_name(self) -> None:
        assert m2.bucket_arn("123456789012") == "arn:aws:s3:::daccord-dev-123456789012"

    def test_role_arn_pattern(self) -> None:
        assert (
            m2.role_arn("123456789012")
            == "arn:aws:iam::123456789012:role/DaccordBedrockBatchService"
        )

    def test_s3_input_prefix_includes_framework_pair_and_model(self) -> None:
        uri = m2.s3_input_prefix("123456789012", "gdpr__pdpa_sg", "llama-4-scout")
        assert uri.startswith("s3://daccord-dev-123456789012/")
        assert "gdpr__pdpa_sg" in uri
        assert "llama-4-scout" in uri

    def test_s3_output_prefix_is_dir_style(self) -> None:
        # Output of `bedrock:CreateModelInvocationJob` is a directory (trailing /),
        # not a single file — Bedrock writes manifest + per-record output files.
        uri = m2.s3_output_prefix("123456789012", "gdpr__pdpa_sg", "llama-4-scout")
        assert uri.endswith("/")


class TestResolveProfile:
    def test_arg_profile_takes_precedence(self, monkeypatch: object) -> None:
        # Even with env var + yaml set, an explicit arg wins.
        os.environ["AWS_PROFILE"] = "env-profile"
        try:
            assert m2.resolve_profile("explicit-arg") == "explicit-arg"
        finally:
            del os.environ["AWS_PROFILE"]

    def test_env_var_used_when_no_arg(self) -> None:
        os.environ["AWS_PROFILE"] = "env-profile"
        try:
            assert m2.resolve_profile(None) == "env-profile"
        finally:
            del os.environ["AWS_PROFILE"]

    def test_yaml_used_when_no_arg_no_env(self, tmp_path: Path) -> None:
        # Inject a fake aws-account.yaml at a controlled path; redirect the
        # module's _repo_root() to return tmp_path.
        (tmp_path / "aws-account.yaml").write_text(
            "profile: yaml-profile\naccount_id: '000000000000'\n",
            encoding="utf-8",
        )
        os.environ.pop("AWS_PROFILE", None)
        with mock.patch.object(m2, "_repo_root", return_value=tmp_path):
            assert m2.resolve_profile(None) == "yaml-profile"

    def test_fallback_when_nothing_resolves(self, tmp_path: Path) -> None:
        # No env var, no yaml file → caravan-poc fallback.
        os.environ.pop("AWS_PROFILE", None)
        with mock.patch.object(m2, "_repo_root", return_value=tmp_path):
            assert m2.resolve_profile(None) == "caravan-poc"

    def test_yaml_without_profile_field_falls_through(self, tmp_path: Path) -> None:
        # aws-account.yaml present but missing `profile:` key → fall to fallback.
        (tmp_path / "aws-account.yaml").write_text(
            "account_id: '000000000000'\nregion: us-east-1\n",
            encoding="utf-8",
        )
        os.environ.pop("AWS_PROFILE", None)
        with mock.patch.object(m2, "_repo_root", return_value=tmp_path):
            assert m2.resolve_profile(None) == "caravan-poc"

    def test_malformed_yaml_falls_through(self, tmp_path: Path) -> None:
        (tmp_path / "aws-account.yaml").write_text(
            "::: this is :::: not yaml ::::\n",
            encoding="utf-8",
        )
        os.environ.pop("AWS_PROFILE", None)
        with mock.patch.object(m2, "_repo_root", return_value=tmp_path):
            assert m2.resolve_profile(None) == "caravan-poc"
