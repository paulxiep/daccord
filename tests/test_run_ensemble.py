"""Tier-7A orchestrator tests — pair enumeration, smoke-mode count, ledger.

These tests cover the pure-logic surface of `scripts/run_ensemble.py` —
no live AWS calls, no boto3 mocking (that's in `tests/test_bedrock_batch.py`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_ensemble  # type: ignore[import-not-found]  # noqa: E402  (sys.path mutated above)


class TestParseFrameworkPair:
    def test_simple_pair(self) -> None:
        assert run_ensemble.parse_framework_pair("gdpr__pdpa_sg") == ("gdpr", "pdpa_sg")

    def test_underscore_in_target_framework(self) -> None:
        # `pdpa_my` is one token even with the embedded `_` — split on LAST `__`.
        assert run_ensemble.parse_framework_pair("gdpr__pdpa_my") == ("gdpr", "pdpa_my")

    def test_underscore_in_source_framework(self) -> None:
        assert run_ensemble.parse_framework_pair("uk_gdpr__bdsg") == ("uk_gdpr", "bdsg")

    def test_underscore_in_both(self) -> None:
        assert run_ensemble.parse_framework_pair("pdpa_my__dpa_2012_ph") == (
            "pdpa_my",
            "dpa_2012_ph",
        )

    def test_malformed_raises(self) -> None:
        with pytest.raises(ValueError, match="malformed"):
            run_ensemble.parse_framework_pair("notvalid")


class TestEnumerateFrameworkPairs:
    def test_full_cross_product_minus_self_pairs(self, tmp_path: Path) -> None:
        # 4-framework fixture → 4*3 = 12 ordered pairs.
        registry_dir = tmp_path / "registry"
        registry_dir.mkdir()
        manifest_lines = [
            json.dumps(
                {
                    "framework": fw,
                    "jurisdiction": jur,
                    "registry_relpath": f"data/registry/{fw}.json",
                    "citation_count": 0,
                    "cites_per_page": None,
                    "toy_gold_recall": None,
                    "toy_gold_missing": [],
                    "sha256_registry": "x",
                    "source_documents": [],
                    "source_sha256": [],
                }
            )
            for fw, jur in [
                ("gdpr", "eu"),
                ("pdpa_sg", "sg"),
                ("pdpa_my", "my"),
                ("bdsg", "de"),
            ]
        ]
        (registry_dir / "manifest.jsonl").write_text(
            "\n".join(manifest_lines) + "\n", encoding="utf-8"
        )
        pairs = run_ensemble.enumerate_framework_pairs(registry_dir)
        assert len(pairs) == 12
        # No self-pairs.
        for pair in pairs:
            src, tgt = pair.split("__", 1)
            assert src != tgt

    def test_nine_frameworks_yield_72_pairs(self, tmp_path: Path) -> None:
        # The real-world case: 9 frameworks → 72 ordered pairs.
        registry_dir = tmp_path / "registry"
        registry_dir.mkdir()
        frameworks = [
            "gdpr",
            "uk_gdpr",
            "loi_il",
            "bdsg",
            "dpa_2018",
            "dpa_2012_ph",
            "pdpa_sg",
            "pdpa_my",
            "pdpa_th",
        ]
        manifest_lines = [
            json.dumps(
                {
                    "framework": fw,
                    "jurisdiction": "xx",
                    "registry_relpath": f"data/registry/{fw}.json",
                    "citation_count": 0,
                    "cites_per_page": None,
                    "toy_gold_recall": None,
                    "toy_gold_missing": [],
                    "sha256_registry": "x",
                    "source_documents": [],
                    "source_sha256": [],
                }
            )
            for fw in frameworks
        ]
        (registry_dir / "manifest.jsonl").write_text(
            "\n".join(manifest_lines) + "\n", encoding="utf-8"
        )
        pairs = run_ensemble.enumerate_framework_pairs(registry_dir)
        assert len(pairs) == 72


class TestLedgerRoundTrip:
    def test_write_and_read_ledger(self, tmp_path: Path) -> None:
        entry = run_ensemble.JobLedgerEntry(
            framework_pair="gdpr__pdpa_sg",
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            job_arn="arn:aws:bedrock:us-east-1:000000000000:job/abc",
            job_name="daccord-gdpr-pdpa-sg-claude",
            status="Submitted",
            smoke=True,
            prompt_count=5,
            input_s3_uri="s3://bucket/ensemble/smoke/in/x.jsonl",
            output_s3_uri="s3://bucket/ensemble/smoke/out/x/",
            submitted_at="2026-05-26T12:00:00+00:00",
        )
        path = tmp_path / "jobs.jsonl"
        run_ensemble.write_ledger(path, [entry])
        loaded = run_ensemble.read_ledger(path)
        assert loaded == [entry]

    def test_empty_ledger_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.jsonl"
        assert run_ensemble.read_ledger(path) == []

    def test_write_is_atomic(self, tmp_path: Path) -> None:
        # Re-running write produces identical content.
        entry = run_ensemble.JobLedgerEntry(
            framework_pair="x__y",
            model="m",
            job_arn="arn",
            job_name="n",
            status="Submitted",
            smoke=False,
            prompt_count=1,
            input_s3_uri="s3://b/in.jsonl",
            output_s3_uri="s3://b/out/",
            submitted_at="2026-05-26T00:00:00+00:00",
        )
        path = tmp_path / "jobs.jsonl"
        run_ensemble.write_ledger(path, [entry])
        content_first = path.read_text(encoding="utf-8")
        run_ensemble.write_ledger(path, [entry])
        content_second = path.read_text(encoding="utf-8")
        assert content_first == content_second


class TestEstimateTokens:
    def test_chars_div_4_heuristic(self) -> None:
        # Match the heuristic used by eval/clients.py.
        prompt = run_ensemble.awsbatch.BatchPrompt(
            record_id="x",
            source_id="x",
            source_jurisdiction="eu",
            source_framework="gdpr",
            source_citation_id="6",
            source_mechanism="body",
            target_jurisdiction="sg",
            target_framework="pdpa_sg",
            system="a" * 40,
            user="b" * 80,
            max_tokens=200,
        )
        in_tok, out_tok = run_ensemble.estimate_tokens_per_prompt(prompt)
        assert in_tok == 30  # 120 chars / 4
        assert out_tok == 200


class TestSmokeConstants:
    def test_smoke_pair_is_gdpr_pdpa_sg(self) -> None:
        assert run_ensemble.SMOKE_FRAMEWORK_PAIR == "gdpr__pdpa_sg"

    def test_smoke_clause_count_is_five(self) -> None:
        assert run_ensemble.SMOKE_CLAUSE_COUNT == 5


class TestBuildSmokePrompts:
    def test_yields_five_gdpr_pdpa_sg_prompts(self, tmp_path: Path) -> None:
        # Use the real toy_v1.jsonl (13 gdpr-source pairs available — verified
        # earlier in TestExtractFrameworkClauses style).
        toy_gold_path = Path(__file__).resolve().parents[1] / "data" / "gold" / "toy_v1.jsonl"
        registry_dir = Path(__file__).resolve().parents[1] / "data" / "registry"
        prompts = run_ensemble.build_smoke_prompts(
            toy_gold_path=toy_gold_path,
            registry_dir=registry_dir,
            max_tokens=256,
        )
        assert len(prompts) == 5
        for p in prompts:
            assert p.source_framework == "gdpr"
            assert p.target_framework == "pdpa_sg"
            assert p.target_jurisdiction == "sg"
            # Registry block must be in the user message (registry-pinned prompt).
            assert "Valid citation_ids for pdpa_sg" in p.user
