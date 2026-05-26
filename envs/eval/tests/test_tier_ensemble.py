"""Tier-6B classifier regression guards.

The HIGH/MED/LOW/SALVAGE rules are the gold-pipeline filter — a misclassification
inflates either the gold pool (false HIGH) or the hand-validation burden (false
LOW). These tests pin each tier transition with the smallest fixture that
exercises the rule.

Tier 7A output schema is also pinned here so a tier-7A breaking change is
caught at this layer (tier-7A code lands next session).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from daccord.ensemble import EnsembleCandidate, TieredPair, classify_tier, tier_framework_pair
from daccord.ensemble.tier import _extract_parent


def _make(model: str, citation_id: str, *, source_id: str = "src-001") -> EnsembleCandidate:
    """Compact fixture builder — only the agreement-relevant fields vary."""
    return EnsembleCandidate(
        source_id=source_id,
        source_jurisdiction="eu",
        source_framework="gdpr",
        source_citation_id="Art. 32",
        source_mechanism="Security of processing.",
        target_jurisdiction="sg",
        target_framework="pdpa_sg",
        model=model,
        citation_id=citation_id,
        target_mechanism="...",
        mapping_justification="...",
    )


class TestParentExtraction:
    def test_simple_id_is_its_own_parent(self) -> None:
        assert _extract_parent("32") == "32"

    def test_alpha_suffix_preserved(self) -> None:
        # 5A and 5 are *different* registry entries in DPA-2018 — don't collapse.
        assert _extract_parent("5A") == "5A"

    def test_subclause_strips_to_parent(self) -> None:
        assert _extract_parent("32(1)") == "32"
        assert _extract_parent("32(1)(a)") == "32"

    def test_thai_id_unchanged(self) -> None:
        assert _extract_parent("มาตรา 19") == "มาตรา 19"

    def test_empty_stays_empty(self) -> None:
        assert _extract_parent("") == ""


class TestClassifyTier:
    def test_high_unanimous(self) -> None:
        cands = [
            _make("llama-4-scout", "24"),
            _make("llama-4-maverick", "24"),
            _make("claude-haiku-4-5", "24"),
            _make("gemini-3-1-flash", "24"),
        ]
        result = classify_tier(cands)
        assert result.tier == "HIGH"
        assert result.consensus_citation_id == "24"
        assert len(result.votes) == 4

    def test_high_unanimous_after_normalization(self) -> None:
        # Section 24, Sec 24, § 24, 24 all normalize to "24" → HIGH.
        cands = [
            _make("llama-4-scout", "Section 24"),
            _make("llama-4-maverick", "Sec 24"),
            _make("claude-haiku-4-5", "§ 24"),
            _make("gemini-3-1-flash", "24"),
        ]
        result = classify_tier(cands)
        assert result.tier == "HIGH"
        assert result.consensus_citation_id == "24"

    def test_med_three_of_four(self) -> None:
        cands = [
            _make("llama-4-scout", "24"),
            _make("llama-4-maverick", "24"),
            _make("claude-haiku-4-5", "24"),
            _make("gemini-3-1-flash", "26"),
        ]
        result = classify_tier(cands)
        assert result.tier == "MED"
        assert result.consensus_citation_id == "24"

    def test_med_parent_agreement_subclause_split(self) -> None:
        # All 4 agreed on Art 32 family but split on sub-clauses — MED via parent.
        cands = [
            _make("llama-4-scout", "32"),
            _make("llama-4-maverick", "32(1)"),
            _make("claude-haiku-4-5", "32(1)(a)"),
            _make("gemini-3-1-flash", "32(2)"),
        ]
        result = classify_tier(cands)
        assert result.tier == "MED"
        assert result.consensus_citation_id == "32"

    def test_low_two_of_four_no_parent_consensus(self) -> None:
        cands = [
            _make("llama-4-scout", "24"),
            _make("llama-4-maverick", "24"),
            _make("claude-haiku-4-5", "26"),
            _make("gemini-3-1-flash", "13"),
        ]
        result = classify_tier(cands)
        assert result.tier == "LOW"
        assert result.consensus_citation_id == "24"  # plurality winner exposed

    def test_low_all_different(self) -> None:
        cands = [
            _make("llama-4-scout", "24"),
            _make("llama-4-maverick", "25"),
            _make("claude-haiku-4-5", "26"),
            _make("gemini-3-1-flash", "27"),
        ]
        result = classify_tier(cands)
        assert result.tier == "LOW"

    def test_salvage_all_empty(self) -> None:
        cands = [
            _make("llama-4-scout", ""),
            _make("llama-4-maverick", ""),
            _make("claude-haiku-4-5", ""),
            _make("gemini-3-1-flash", ""),
        ]
        result = classify_tier(cands)
        assert result.tier == "SALVAGE"
        assert result.consensus_citation_id == ""

    def test_salvage_three_empty_one_dissenting(self) -> None:
        # 3/4 say "no analog", 1 dissents — SALVAGE (the lone vote is suspect).
        cands = [
            _make("llama-4-scout", ""),
            _make("llama-4-maverick", ""),
            _make("claude-haiku-4-5", ""),
            _make("gemini-3-1-flash", "24"),
        ]
        result = classify_tier(cands)
        assert result.tier == "SALVAGE"

    def test_high_when_two_empty_two_agree_falls_to_low(self) -> None:
        # 2 empty + 2 agreeing on "24" — not enough empties for SALVAGE,
        # not majority of non-empty for MED; agreement is 2/4 → LOW.
        cands = [
            _make("llama-4-scout", "24"),
            _make("llama-4-maverick", "24"),
            _make("claude-haiku-4-5", ""),
            _make("gemini-3-1-flash", ""),
        ]
        result = classify_tier(cands)
        assert result.tier == "LOW"

    def test_inconsistent_source_id_raises(self) -> None:
        cands = [
            _make("llama-4-scout", "24", source_id="src-A"),
            _make("llama-4-maverick", "24", source_id="src-B"),
        ]
        with pytest.raises(ValueError, match="inconsistent source_id"):
            classify_tier(cands)

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="empty candidates"):
            classify_tier([])

    def test_parse_error_preserved_in_vote(self) -> None:
        cands = [
            EnsembleCandidate(
                source_id="src-001",
                source_jurisdiction="eu",
                source_framework="gdpr",
                source_citation_id="Art. 32",
                source_mechanism="Security of processing.",
                target_jurisdiction="sg",
                target_framework="pdpa_sg",
                model="local-qwen-7b",
                citation_id="",
                target_mechanism="",
                mapping_justification="",
                parse_error="json.JSONDecodeError at char 0",
            ),
            _make("llama-4-scout", "24"),
            _make("llama-4-maverick", "24"),
            _make("gemini-3-1-flash", "24"),
        ]
        result = classify_tier(cands)
        # 3 agree, 1 empty (parse error) → MED
        assert result.tier == "MED"
        assert result.consensus_citation_id == "24"
        # The error-bearing vote is preserved
        errored = [v for v in result.votes if v.parse_error is not None]
        assert len(errored) == 1
        assert errored[0].model == "local-qwen-7b"


class TestThreeModelEnsemble:
    """Tests for the documented N=3 fallback (if Bedrock model access stalls
    for one F9 seat, tier 8 retiers HIGH=3/3, MED=2/3, LOW=≤1/3)."""

    def test_high_three_of_three(self) -> None:
        cands = [
            _make("llama-4-scout", "24"),
            _make("llama-4-maverick", "24"),
            _make("claude-haiku-4-5", "24"),
        ]
        result = classify_tier(cands)
        assert result.tier == "HIGH"

    def test_med_two_of_three(self) -> None:
        cands = [
            _make("llama-4-scout", "24"),
            _make("llama-4-maverick", "24"),
            _make("claude-haiku-4-5", "26"),
        ]
        result = classify_tier(cands)
        assert result.tier == "MED"
        assert result.consensus_citation_id == "24"

    def test_low_no_agreement_three(self) -> None:
        cands = [
            _make("llama-4-scout", "24"),
            _make("llama-4-maverick", "25"),
            _make("claude-haiku-4-5", "26"),
        ]
        result = classify_tier(cands)
        assert result.tier == "LOW"


class TestTierFrameworkPair:
    def _write_jsonl(self, path: Path, candidates: list[EnsembleCandidate]) -> None:
        path.write_text(
            "\n".join(c.model_dump_json() for c in candidates) + "\n",
            encoding="utf-8",
        )

    def test_loads_four_model_pair_and_tiers(self, tmp_path: Path) -> None:
        fp = "gdpr__pdpa_sg"
        models = ["llama-4-scout", "llama-4-maverick", "claude-haiku-4-5", "gemini-3-1-flash"]
        # Two source clauses, one HIGH-tier, one LOW-tier.
        for m in models:
            cands_for_model = [
                _make(m, "24", source_id="src-001"),  # all agree → HIGH
                _make(m, f"{20 + models.index(m)}", source_id="src-002"),  # all disagree → LOW
            ]
            self._write_jsonl(tmp_path / f"{fp}__{m}.jsonl", cands_for_model)

        results = tier_framework_pair(framework_pair=fp, raw_dir=tmp_path)
        assert len(results) == 2
        by_id = {r.source_id: r for r in results}
        assert by_id["src-001"].tier == "HIGH"
        assert by_id["src-002"].tier == "LOW"

    def test_expected_models_guard_catches_partial_run(self, tmp_path: Path) -> None:
        fp = "gdpr__pdpa_sg"
        # Only 3 of 4 models contributed for src-001.
        for m in ["llama-4-scout", "llama-4-maverick", "claude-haiku-4-5"]:
            self._write_jsonl(
                tmp_path / f"{fp}__{m}.jsonl",
                [_make(m, "24", source_id="src-001")],
            )
        with pytest.raises(ValueError, match="missing votes"):
            tier_framework_pair(
                framework_pair=fp,
                raw_dir=tmp_path,
                expected_models=[
                    "llama-4-scout",
                    "llama-4-maverick",
                    "claude-haiku-4-5",
                    "gemini-3-1-flash",
                ],
            )

    def test_no_files_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            tier_framework_pair(framework_pair="nope__nope", raw_dir=tmp_path)


class TestTieredPairSerialization:
    def test_roundtrip(self) -> None:
        cands = [
            _make("llama-4-scout", "24"),
            _make("llama-4-maverick", "24"),
            _make("claude-haiku-4-5", "24"),
            _make("gemini-3-1-flash", "24"),
        ]
        tp = classify_tier(cands)
        as_json = tp.model_dump_json()
        restored = TieredPair.model_validate_json(as_json)
        assert restored == tp
