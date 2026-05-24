"""Eval-harness inference-shape tests.

`CitationCandidate`, `ModelResponse`, `PromptMessages` are tier-2B-specific
inference-time data shapes. The gold-set shapes (`GoldPair`/`GoldSet`)
moved to [daccord.gold]; see [test_gold_schema.py] for their tests.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from daccord.eval.schema import CitationCandidate, ModelResponse, PromptMessages


class TestInferenceShapes:
    def test_citation_candidate_validates(self) -> None:
        c = CitationCandidate(
            citation_id="Section 24",
            target_mechanism="Reasonable security arrangements.",
            mapping_justification="Both establish a general security obligation.",
        )
        assert c.citation_id == "Section 24"

    def test_model_response_candidates_default_none(self) -> None:
        resp = ModelResponse(
            model="claude-3-5-sonnet-20241022",
            top1=CitationCandidate(
                citation_id="Section 24",
                target_mechanism="...",
                mapping_justification="...",
            ),
            raw_text="{...}",
            input_tokens=120,
            output_tokens=80,
            latency_ms=1234.5,
        )
        assert resp.candidates is None
        assert resp.parse_error is None

    def test_model_response_parse_error_allows_null_top1(self) -> None:
        resp = ModelResponse(
            model="qwen-7b-local",
            top1=None,
            raw_text="not json at all",
            input_tokens=120,
            output_tokens=10,
            latency_ms=850.0,
            parse_error="json.decoder.JSONDecodeError at char 0",
        )
        assert resp.top1 is None
        assert resp.parse_error is not None

    def test_prompt_messages_requires_both_roles(self) -> None:
        msg = PromptMessages(system="You are ...", user="Map this clause: ...")
        assert msg.system.startswith("You are")
        with pytest.raises(ValidationError):
            PromptMessages(system="only system")  # type: ignore[call-arg]
