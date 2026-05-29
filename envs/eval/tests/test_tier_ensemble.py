"""Tier-6B fuzzy classifier regression guards.

The fuzzy `agreement_score` + `valid_vote_count` fields are the primary
output; the legacy HIGH/MED/LOW/SALVAGE bucket is derived for back-compat.
Each test pins both the new fuzzy fields AND the derived bucket so a
miscomputation in either layer is caught.

Tier 7A output schema (`EnsembleCandidate`) is also pinned here so a
tier-7A breaking change is caught at this layer.
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


class TestClassifyTierFuzzyFields:
    """Pin the three new fuzzy fields on representative cases."""

    def test_high_unanimous_populates_fuzzy_fields(self) -> None:
        cands = [
            _make("llama-4-scout", "24"),
            _make("llama-4-maverick", "24"),
            _make("claude-haiku-4-5", "24"),
            _make("gemini-3-1-flash", "24"),
        ]
        result = classify_tier(cands)
        assert result.tier == "HIGH"
        assert result.consensus_citation_id == "24"
        assert result.valid_vote_count == 4
        assert result.consensus_vote_count == 4
        assert result.agreement_score == 1.0
        assert len(result.votes) == 4

    def test_med_three_of_four_score_is_quarter_fractions(self) -> None:
        cands = [
            _make("llama-4-scout", "24"),
            _make("llama-4-maverick", "24"),
            _make("claude-haiku-4-5", "24"),
            _make("gemini-3-1-flash", "26"),
        ]
        result = classify_tier(cands)
        assert result.tier == "MED"
        assert result.consensus_citation_id == "24"
        assert result.valid_vote_count == 4
        assert result.consensus_vote_count == 3
        assert result.agreement_score == 0.75

    def test_low_plurality_two_of_four_score_is_half(self) -> None:
        cands = [
            _make("llama-4-scout", "24"),
            _make("llama-4-maverick", "24"),
            _make("claude-haiku-4-5", "26"),
            _make("gemini-3-1-flash", "13"),
        ]
        result = classify_tier(cands)
        assert result.tier == "LOW"
        assert result.consensus_citation_id == "24"
        assert result.valid_vote_count == 4
        assert result.consensus_vote_count == 2
        assert result.agreement_score == 0.5

    def test_salvage_uses_sentinel_score_negative_one(self) -> None:
        cands = [_make(m, "") for m in ("a", "b", "c", "d")]
        result = classify_tier(cands)
        assert result.tier == "SALVAGE"
        assert result.consensus_citation_id == ""
        assert result.valid_vote_count == 0
        assert result.consensus_vote_count == 0
        assert result.agreement_score == -1.0


class TestClassifyTierBucketBoundaries:
    """Pin the legacy HIGH/MED/LOW/SALVAGE bucket boundaries under fuzzy logic."""

    def test_high_unanimous(self) -> None:
        cands = [
            _make("llama-4-scout", "24"),
            _make("llama-4-maverick", "24"),
            _make("claude-haiku-4-5", "24"),
            _make("gemini-3-1-flash", "24"),
        ]
        assert classify_tier(cands).tier == "HIGH"

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

    def test_subclause_split_unanimous_parent_is_HIGH_via_parent_fallback(self) -> None:
        # All 4 agree on Art 32 family but split on sub-clauses → parent
        # fallback fires; consensus = "32", score = 1.0, derived HIGH.
        #
        # (Pre-fuzzy this was MED. Under fuzzy, unanimous parent agreement
        # is full agreement at the parent level: hand-val refines sub-clause.)
        cands = [
            _make("llama-4-scout", "32"),
            _make("llama-4-maverick", "32(1)"),
            _make("claude-haiku-4-5", "32(1)(a)"),
            _make("gemini-3-1-flash", "32(2)"),
        ]
        result = classify_tier(cands)
        assert result.tier == "HIGH"
        assert result.consensus_citation_id == "32"
        assert result.valid_vote_count == 4
        assert result.consensus_vote_count == 4
        assert result.agreement_score == 1.0

    def test_majority_at_citation_overrules_parent_fallback(self) -> None:
        # 3/4 agree on "32(1)" exactly; 1 says "32(2)". Citation-level
        # majority wins over parent fallback — consensus = "32(1)" (more
        # specific), not "32" (parent). Score = 0.75 → derived MED.
        cands = [
            _make("llama-4-scout", "32(1)"),
            _make("llama-4-maverick", "32(1)"),
            _make("claude-haiku-4-5", "32(1)"),
            _make("gemini-3-1-flash", "32(2)"),
        ]
        result = classify_tier(cands)
        assert result.tier == "MED"
        assert result.consensus_citation_id == "32(1)"
        assert result.agreement_score == 0.75

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
        assert classify_tier(cands).tier == "LOW"

    def test_salvage_all_empty(self) -> None:
        cands = [_make(m, "") for m in ("a", "b", "c", "d")]
        result = classify_tier(cands)
        assert result.tier == "SALVAGE"
        assert result.consensus_citation_id == ""

    def test_three_empty_one_dissenting_is_LOW_under_fuzzy(self) -> None:
        # 3/4 say "no analog", 1 dissents — under fuzzy, valid_vote_count = 1,
        # score = 1.0, but valid_vote_count < 2 so derives LOW (not HIGH).
        # The lone dissenter is hand-validatable, not salvageable.
        #
        # (Pre-fuzzy this was SALVAGE via the "≥ N-1 empty" rule.)
        cands = [
            _make("llama-4-scout", ""),
            _make("llama-4-maverick", ""),
            _make("claude-haiku-4-5", ""),
            _make("gemini-3-1-flash", "24"),
        ]
        result = classify_tier(cands)
        assert result.tier == "LOW"
        assert result.consensus_citation_id == "24"
        assert result.valid_vote_count == 1
        assert result.consensus_vote_count == 1
        assert result.agreement_score == 1.0

    def test_two_empty_two_agree_is_HIGH_under_fuzzy(self) -> None:
        # The design-change-motivating case: 2 valid votes agree, 2 missed.
        # Under fuzzy, denominator is valid_vote_count = 2, not N = 4, so
        # this is full agreement → HIGH (not LOW as pre-fuzzy classified).
        cands = [
            _make("llama-4-scout", "24"),
            _make("llama-4-maverick", "24"),
            _make("claude-haiku-4-5", ""),
            _make("gemini-3-1-flash", ""),
        ]
        result = classify_tier(cands)
        assert result.tier == "HIGH"
        assert result.consensus_citation_id == "24"
        assert result.valid_vote_count == 2
        assert result.consensus_vote_count == 2
        assert result.agreement_score == 1.0

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

    def test_parse_error_treated_as_missing_vote(self) -> None:
        # 3 valid votes agree; 1 parse_error reduces valid_vote_count to 3,
        # not 4. Score = 3/3 = 1.0 → HIGH. (Pre-fuzzy this was MED.)
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
        assert result.tier == "HIGH"
        assert result.consensus_citation_id == "24"
        assert result.valid_vote_count == 3
        assert result.consensus_vote_count == 3
        assert result.agreement_score == 1.0
        # The error-bearing vote is preserved
        errored = [v for v in result.votes if v.parse_error is not None]
        assert len(errored) == 1
        assert errored[0].model == "local-qwen-7b"

    def test_all_parse_error_is_SALVAGE_with_sentinel(self) -> None:
        # All 4 parse_error — valid_vote_count = 0, sentinel score -1.0.
        cands = [
            EnsembleCandidate(
                source_id="src-001",
                source_jurisdiction="eu",
                source_framework="gdpr",
                source_citation_id="Art. 32",
                source_mechanism="Security of processing.",
                target_jurisdiction="sg",
                target_framework="pdpa_sg",
                model=m,
                citation_id="",
                target_mechanism="",
                mapping_justification="",
                parse_error="context_length_exceeded",
            )
            for m in ("a", "b", "c", "d")
        ]
        result = classify_tier(cands)
        assert result.tier == "SALVAGE"
        assert result.valid_vote_count == 0
        assert result.agreement_score == -1.0


class TestThreeModelEnsemble:
    """N=3 fallback (if one F9 seat's access stalls). Same fuzzy rules,
    smaller denominator."""

    def test_high_three_of_three(self) -> None:
        cands = [
            _make("llama-4-scout", "24"),
            _make("llama-4-maverick", "24"),
            _make("claude-haiku-4-5", "24"),
        ]
        result = classify_tier(cands)
        assert result.tier == "HIGH"
        assert result.agreement_score == 1.0
        assert result.valid_vote_count == 3

    def test_med_two_of_three(self) -> None:
        cands = [
            _make("llama-4-scout", "24"),
            _make("llama-4-maverick", "24"),
            _make("claude-haiku-4-5", "26"),
        ]
        result = classify_tier(cands)
        assert result.tier == "MED"
        assert result.consensus_citation_id == "24"
        assert result.consensus_vote_count == 2
        assert result.agreement_score == pytest.approx(2 / 3)

    def test_low_no_agreement_three(self) -> None:
        cands = [
            _make("llama-4-scout", "24"),
            _make("llama-4-maverick", "25"),
            _make("claude-haiku-4-5", "26"),
        ]
        result = classify_tier(cands)
        assert result.tier == "LOW"
        assert result.agreement_score == pytest.approx(1 / 3)


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


class TestExtraDirsGlob:
    """Tier 6B++ — `tier_framework_pair` reads both raw/ and raw_local/ dirs.

    The RAG seat (5th-seat strategy) writes to data/ensemble/raw_local/ to
    keep paid-API raw/ files untouched. Its votes are preserved in
    `TieredPair.votes` for downstream consumption but treated as side-info:
    `valid_vote_count` / `agreement_score` / `tier` are computed over LLM
    seats only. The local seat's signal is surfaced via the derived
    `rag_concurs` + `rag_vote_citation_id` fields for downstream opt-in
    promotion (splits + labeler).
    """

    def _write_jsonl(self, path: Path, candidates: list[EnsembleCandidate]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(c.model_dump_json() for c in candidates) + "\n",
            encoding="utf-8",
        )

    def test_rag_concurs_sets_flag_but_does_not_change_tier(self, tmp_path: Path) -> None:
        """3 LLMs HIGH + RAG agrees: LLM-denominator stays N=3; rag_concurs=True."""
        raw = tmp_path / "raw"
        raw_local = tmp_path / "raw_local"
        fp = "gdpr__pdpa_sg"

        for m in ("llama-4-scout", "llama-4-maverick", "claude-haiku-4-5"):
            self._write_jsonl(raw / f"{fp}__{m}.jsonl", [_make(m, "24")])
        self._write_jsonl(
            raw_local / f"{fp}__local-rag-mpnet.jsonl",
            [_make("local-rag-mpnet", "24")],
        )

        r = tier_framework_pair(framework_pair=fp, raw_dir=raw, extra_dirs=[raw_local])[0]
        assert r.tier == "HIGH"
        assert r.valid_vote_count == 3  # LLM-only denominator
        assert r.consensus_vote_count == 3
        assert r.agreement_score == 1.0
        assert r.rag_concurs is True
        assert r.rag_vote_citation_id == "24"
        # The RAG vote is preserved in the votes list for the labeler.
        models = {v.model for v in r.votes}
        assert "local-rag-mpnet" in models
        assert len(r.votes) == 4  # 3 LLM + 1 RAG

    def test_rag_dissent_does_NOT_demote_HIGH(self, tmp_path: Path) -> None:
        """4/4 LLM unanimous HIGH stays HIGH even when RAG dissents.

        RAG is informational, not a standard-raiser — 4 LLM agreement is
        already high confidence; a dissenting retrieval vote shouldn't
        downgrade it. `rag_concurs=False` records the disagreement for
        the labeler / future spot-check workflows.
        """
        raw = tmp_path / "raw"
        raw_local = tmp_path / "raw_local"
        fp = "gdpr__pdpa_sg"

        for m in ("llama-4-scout", "llama-4-maverick", "claude-haiku-4-5", "gemini-3-1-flash"):
            self._write_jsonl(raw / f"{fp}__{m}.jsonl", [_make(m, "24")])
        self._write_jsonl(
            raw_local / f"{fp}__local-rag-mpnet.jsonl", [_make("local-rag-mpnet", "26")]
        )

        r = tier_framework_pair(framework_pair=fp, raw_dir=raw, extra_dirs=[raw_local])[0]
        assert r.tier == "HIGH"
        assert r.valid_vote_count == 4
        assert r.consensus_vote_count == 4
        assert r.consensus_citation_id == "24"
        assert r.agreement_score == 1.0
        assert r.rag_concurs is False
        assert r.rag_vote_citation_id == "26"

    def test_rag_concurs_on_MED_eligible_for_promotion(self, tmp_path: Path) -> None:
        """3/4 LLM MED + RAG concurs with consensus → rag_concurs=True.

        Splits' opt-in RAG-promotion path uses this flag (mirror of
        bidirectional-consistent promotion).
        """
        raw = tmp_path / "raw"
        raw_local = tmp_path / "raw_local"
        fp = "gdpr__pdpa_sg"

        for m in ("llama-4-scout", "llama-4-maverick", "claude-haiku-4-5"):
            self._write_jsonl(raw / f"{fp}__{m}.jsonl", [_make(m, "24")])
        self._write_jsonl(raw / f"{fp}__gemini-3-1-flash.jsonl", [_make("gemini-3-1-flash", "26")])
        self._write_jsonl(
            raw_local / f"{fp}__local-rag-mpnet.jsonl",
            [_make("local-rag-mpnet", "24")],
        )

        r = tier_framework_pair(framework_pair=fp, raw_dir=raw, extra_dirs=[raw_local])[0]
        assert r.tier == "MED"
        assert r.valid_vote_count == 4
        assert r.consensus_vote_count == 3
        assert r.consensus_citation_id == "24"
        assert r.rag_concurs is True
        assert r.rag_vote_citation_id == "24"

    def test_rag_alone_cannot_create_gold_from_salvage(self, tmp_path: Path) -> None:
        """All 4 LLMs empty + RAG retrieves something → still SALVAGE.

        RAG is informational. A single retrieval hit when every LLM said
        "no analog" is insufficient signal for gold — tier stays SALVAGE;
        the RAG vote is preserved for human review.
        """
        raw = tmp_path / "raw"
        raw_local = tmp_path / "raw_local"
        fp = "gdpr__pdpa_sg"

        for m in ("llama-4-scout", "llama-4-maverick", "claude-haiku-4-5", "gemini-3-1-flash"):
            self._write_jsonl(raw / f"{fp}__{m}.jsonl", [_make(m, "")])
        self._write_jsonl(
            raw_local / f"{fp}__local-rag-mpnet.jsonl",
            [_make("local-rag-mpnet", "24")],
        )

        r = tier_framework_pair(framework_pair=fp, raw_dir=raw, extra_dirs=[raw_local])[0]
        assert r.tier == "SALVAGE"
        assert r.valid_vote_count == 0  # LLM denominator only
        assert r.agreement_score == -1.0
        assert r.rag_vote_citation_id == "24"  # preserved for labeler
        # rag_concurs False because consensus_citation_id is "".
        assert r.rag_concurs is False

    def test_extra_dirs_absent_falls_back_to_raw_only(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        fp = "gdpr__pdpa_sg"
        for m in ("llama-4-scout", "llama-4-maverick", "claude-haiku-4-5", "gemini-3-1-flash"):
            self._write_jsonl(raw / f"{fp}__{m}.jsonl", [_make(m, "24")])

        r = tier_framework_pair(framework_pair=fp, raw_dir=raw, extra_dirs=[tmp_path / "nope"])[0]
        assert r.tier == "HIGH"
        assert r.valid_vote_count == 4  # only the 4 LLM seats
        assert r.rag_concurs is False
        assert r.rag_vote_citation_id == ""


class TestTieredPairSerialization:
    def test_roundtrip_carries_fuzzy_fields(self) -> None:
        cands = [
            _make("llama-4-scout", "24"),
            _make("llama-4-maverick", "24"),
            _make("claude-haiku-4-5", "24"),
            _make("gemini-3-1-flash", "26"),
        ]
        tp = classify_tier(cands)
        as_json = tp.model_dump_json()
        restored = TieredPair.model_validate_json(as_json)
        assert restored == tp
        # Spot-check the new fields survived the roundtrip.
        assert restored.agreement_score == 0.75
        assert restored.valid_vote_count == 4
        assert restored.consensus_vote_count == 3
