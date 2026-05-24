"""Prompt regression guard for tier 2B.

A prompt change is a baseline-comparability event — any edit to the eval
generator or judge prompt between M0 and M4 invalidates the per-model
baseline. These tests pin the prompt's load-bearing structure so a
non-deliberate edit fails CI before it ships.
"""

from __future__ import annotations

from daccord.eval.prompts import (
    EVAL_SYSTEM,
    EVAL_USER_TEMPLATE,
    JUDGE_SYSTEM,
    JUDGE_USER_TEMPLATE,
    build_eval_prompt,
    build_judge_prompt,
)
from daccord.eval.schema import CitationCandidate
from daccord.gold import GoldPair

GOLD = GoldPair(
    id="tg-001",
    source_jurisdiction="eu",
    source_framework="gdpr",
    source_citation_id="Art. 32",
    source_mechanism="Security of processing; appropriate technical and organisational measures.",
    source_language="en",
    target_jurisdiction="sg",
    target_framework="pdpa_sg",
    target_citation_id="Section 24",
    target_mechanism="Protection of personal data with reasonable security arrangements.",
    target_language="en",
    notes=None,
)


class TestEvalPrompt:
    def test_system_pins_json_schema(self) -> None:
        for field in ("citation_id", "target_mechanism", "mapping_justification"):
            assert field in EVAL_SYSTEM

    def test_system_pins_no_analogue_escape_hatch(self) -> None:
        assert "no analogous clause" in EVAL_SYSTEM
        assert "empty string" in EVAL_SYSTEM

    def test_user_template_includes_all_source_target_fields(self) -> None:
        for placeholder in (
            "{source_jurisdiction}",
            "{source_framework}",
            "{source_citation_id}",
            "{source_mechanism}",
            "{target_jurisdiction}",
            "{target_framework}",
        ):
            assert placeholder in EVAL_USER_TEMPLATE

    def test_user_template_omits_target_citation_id(self) -> None:
        # The gold target_citation_id MUST NOT leak into the eval prompt —
        # that would make the task trivial (give the model the answer).
        assert "{target_citation_id}" not in EVAL_USER_TEMPLATE
        assert "{target_mechanism}" not in EVAL_USER_TEMPLATE

    def test_build_eval_prompt_renders(self) -> None:
        msgs = build_eval_prompt(GOLD)
        assert msgs.system == EVAL_SYSTEM
        assert "Art. 32" in msgs.user
        assert "pdpa_sg" in msgs.user
        # Sanity: gold answer is NOT in the rendered prompt
        assert "Section 24" not in msgs.user
        assert "reasonable security arrangements" not in msgs.user


class TestJudgePrompt:
    def test_system_pins_judge_schema(self) -> None:
        for field in ("score", "bucket", "reasoning"):
            assert field in JUDGE_SYSTEM

    def test_system_pins_buckets(self) -> None:
        for bucket in (
            "wrong",
            "partial_wrong",
            "partial_right",
            "substantively_right",
            "exact",
        ):
            assert bucket in JUDGE_SYSTEM

    def test_system_pins_anchor_descriptions(self) -> None:
        for anchor in ("0.0", "0.25", "0.5", "0.75", "1.0"):
            assert anchor in JUDGE_SYSTEM

    def test_user_template_shows_gold_and_candidate(self) -> None:
        for placeholder in (
            "{expected_citation_id}",
            "{expected_mechanism}",
            "{predicted_citation_id}",
            "{predicted_mechanism}",
            "{predicted_justification}",
        ):
            assert placeholder in JUDGE_USER_TEMPLATE

    def test_build_judge_prompt_renders(self) -> None:
        candidate = CitationCandidate(
            citation_id="Section 24",
            target_mechanism="Reasonable security measures.",
            mapping_justification="Both require appropriate security.",
        )
        msgs = build_judge_prompt(GOLD, candidate)
        assert msgs.system == JUDGE_SYSTEM
        assert "Section 24" in msgs.user
        assert "Reasonable security measures" in msgs.user
        assert "Art. 32" in msgs.user

    def test_build_judge_prompt_handles_empty_candidate_fields(self) -> None:
        candidate = CitationCandidate(
            citation_id="",
            target_mechanism="",
            mapping_justification="",
        )
        msgs = build_judge_prompt(GOLD, candidate)
        assert "<no citation returned>" in msgs.user
        assert "<no mechanism returned>" in msgs.user
        assert "<no justification returned>" in msgs.user
