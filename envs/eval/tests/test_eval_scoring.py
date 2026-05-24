"""Scoring + judge + aggregation tests.

`normalize_citation_id` is the Tier-1 contract — its table is the single
biggest determinant of baseline numbers. Test it exhaustively here so a
casual edit fails CI loudly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from daccord.costs.config import (
    CONFIG_PATH_ENV,
    DAILY_CSV_PATH_ENV,
    INFLIGHT_PATH_ENV,
    Provider,
)
from daccord.eval.schema import CitationCandidate, ModelResponse, PromptMessages
from daccord.eval.scoring import (
    EvalRow,
    JudgeScore,
    _parse_judge,
    aggregate_rows,
    bucket_counts,
    build_eval_row,
    citation_match_top1,
    citation_match_topk,
    judge_pair,
    judge_pairs,
    normalize_citation_id,
)
from daccord.gold import GoldPair

COSTS_TOML = """\
warning_threshold_usd = 25.0
consecutive_days_for_alert = 2

[caps_usd_per_day]
anthropic = 30.0

[caps_requests_per_day]
groq = 14400
google_gemini = 1500
"""


@pytest.fixture
def costs_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cf = tmp_path / "config.toml"
    cf.write_text(COSTS_TOML, encoding="utf-8")
    monkeypatch.setenv(CONFIG_PATH_ENV, str(cf))
    monkeypatch.setenv(INFLIGHT_PATH_ENV, str(tmp_path / "inflight.sqlite"))
    monkeypatch.setenv(DAILY_CSV_PATH_ENV, str(tmp_path / "daily.csv"))
    monkeypatch.delenv("DACCORD_COSTS_OVERRIDE", raising=False)
    return tmp_path


GOLD = GoldPair(
    id="tg-001",
    source_jurisdiction="eu",
    source_framework="gdpr",
    source_citation_id="Art. 32",
    source_mechanism="Security of processing.",
    source_language="en",
    target_jurisdiction="sg",
    target_framework="pdpa_sg",
    target_citation_id="Section 24",
    target_mechanism="Reasonable security arrangements.",
    target_language="en",
    notes=None,
)


class TestNormalization:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Art. 32", "32"),
            ("Article 32", "32"),
            ("32", "32"),
            ("§32", "32"),
            ("Section 32", "32"),
            ("section 32", "32"),
            ("Sec. 32", "32"),
            ("32(1)", "32(1)"),
            ("32 (1)", "32(1)"),
            ("32 1 a", "32(1)(a)"),
            ("32(1)(a)", "32(1)(a)"),
            ("  Art. 32  ", "32"),
            ("", ""),
        ],
    )
    def test_table(self, raw: str, expected: str) -> None:
        assert normalize_citation_id(raw) == expected

    def test_distinct_ids_stay_distinct(self) -> None:
        assert normalize_citation_id("33") != normalize_citation_id("32")
        assert normalize_citation_id("Art. 33") != normalize_citation_id("Art. 32")


class TestCitationMatchTop1:
    def test_match_across_format_variants(self) -> None:
        assert citation_match_top1("Art. 32", "Article 32") is True
        assert citation_match_top1("32(1)", "32 (1)") is True

    def test_mismatch_when_numbers_differ(self) -> None:
        assert citation_match_top1("Art. 32", "Art. 33") is False

    def test_empty_strings_never_match(self) -> None:
        assert citation_match_top1("", "32") is False
        assert citation_match_top1("32", "") is False


class TestCitationMatchTopK:
    def test_any_candidate_matches(self) -> None:
        assert citation_match_topk(["Art. 33", "Section 24"], "Sec. 24") is True

    def test_no_match_when_all_wrong(self) -> None:
        assert citation_match_topk(["Art. 33", "Art. 34"], "Art. 32") is False

    def test_empty_list_never_matches(self) -> None:
        assert citation_match_topk([], "32") is False


class TestParseJudge:
    def test_valid_payload(self) -> None:
        payload = {"score": 0.8, "bucket": "substantively_right", "reasoning": "matches"}
        s = _parse_judge(json.dumps(payload), "gemini-2.5-flash")
        assert s.score == pytest.approx(0.8)
        assert s.bucket == "substantively_right"
        assert s.parse_error is None

    def test_out_of_range_score_clipped(self) -> None:
        s = _parse_judge(json.dumps({"score": 1.5, "bucket": "exact", "reasoning": "ok"}), "j")
        assert s.score == 1.0
        s2 = _parse_judge(json.dumps({"score": -0.2, "bucket": "wrong", "reasoning": "ok"}), "j")
        assert s2.score == 0.0

    def test_unknown_bucket_falls_back_to_wrong(self) -> None:
        s = _parse_judge(
            json.dumps({"score": 0.5, "bucket": "almost_right", "reasoning": "huh"}), "j"
        )
        assert s.bucket == "wrong"

    def test_malformed_json_emits_sentinel(self) -> None:
        s = _parse_judge("not json", "j")
        assert s.score == 0.0
        assert s.bucket == "wrong"
        assert s.parse_error is not None

    def test_nan_score_coerced_to_zero(self) -> None:
        payload = json.dumps({"score": float("nan"), "bucket": "wrong", "reasoning": ""})
        s = _parse_judge(payload, "j")
        assert s.score == 0.0


class _FakeJudge:
    provider: Provider = "google_gemini"

    def __init__(self, score: float, model: str = "fake-judge") -> None:
        self.model = model
        self._score = score
        self.calls = 0

    def judge(self, messages: PromptMessages, *, run_id: str, batch_id: str) -> JudgeScore:
        self.calls += 1
        return JudgeScore(
            score=self._score,
            bucket="substantively_right" if self._score >= 0.7 else "partial_right",
            reasoning=f"fake judge call {self.calls}",
            judge_model=self.model,
        )


def _make_response(
    citation: str | None = "Section 24", parse_error: str | None = None
) -> ModelResponse:
    top1: CitationCandidate | None = None
    if citation is not None:
        top1 = CitationCandidate(
            citation_id=citation,
            target_mechanism="m",
            mapping_justification="j",
        )
    return ModelResponse(
        model="fake-gen",
        top1=top1,
        raw_text="{}",
        input_tokens=10,
        output_tokens=10,
        latency_ms=1.0,
        parse_error=parse_error,
    )


class TestJudgePair:
    def test_normal_path_invokes_judge(self) -> None:
        j = _FakeJudge(0.8)
        s = judge_pair(GOLD, _make_response(), j, run_id="r", batch_id="b")
        assert j.calls == 1
        assert s.score == 0.8

    def test_generator_parse_failure_short_circuits_with_zero(self) -> None:
        j = _FakeJudge(0.9)
        resp = _make_response(citation=None, parse_error="json decode at char 0")
        s = judge_pair(GOLD, resp, j, run_id="r", batch_id="b")
        assert j.calls == 0  # judge not called when no candidate
        assert s.score == 0.0
        assert "generator parse failure" in s.reasoning

    def test_judge_pairs_loops_single_call(self) -> None:
        j = _FakeJudge(0.5)
        pairs = [(GOLD, _make_response()), (GOLD, _make_response("Section 25"))]
        scores = judge_pairs(pairs, j, run_id="r", batch_id="b")
        assert len(scores) == 2
        assert j.calls == 2


class TestEvalRow:
    def test_build_eval_row_match(self) -> None:
        s = JudgeScore(score=0.9, bucket="exact", reasoning="ok", judge_model="fake")
        row = build_eval_row(GOLD, _make_response("Section 24"), s)
        assert row.citation_match == 1
        assert row.predicted_citation_id == "Section 24"
        assert row.expected_citation_id == "Section 24"
        assert row.judge_bucket == "exact"

    def test_build_eval_row_mismatch(self) -> None:
        s = JudgeScore(score=0.1, bucket="wrong", reasoning="no", judge_model="fake")
        row = build_eval_row(GOLD, _make_response("Section 25"), s)
        assert row.citation_match == 0

    def test_build_eval_row_format_variant_matches(self) -> None:
        # "Sec. 24" should normalize to match "Section 24"
        s = JudgeScore(score=0.9, bucket="exact", reasoning="ok", judge_model="fake")
        row = build_eval_row(GOLD, _make_response("Sec. 24"), s)
        assert row.citation_match == 1

    def test_build_eval_row_parse_failure(self) -> None:
        s = JudgeScore(score=0.0, bucket="wrong", reasoning="gen failed", judge_model="fake")
        row = build_eval_row(GOLD, _make_response(citation=None, parse_error="bad"), s)
        assert row.citation_match == 0
        assert row.predicted_citation_id == ""


def _row(
    gold_id: str = "g1",
    *,
    model: str = "fake",
    target_jurisdiction: str = "sg",
    target_language: str = "en",
    source_framework: str = "gdpr",
    target_framework: str = "pdpa_sg",
    citation_match: int = 1,
    judge_score: float = 0.9,
    judge_bucket: str = "exact",
) -> EvalRow:
    return EvalRow(
        gold_id=gold_id,
        model=model,
        source_jurisdiction="eu",
        source_framework=source_framework,
        target_jurisdiction=target_jurisdiction,
        target_framework=target_framework,
        source_language="en",
        target_language=target_language,
        predicted_citation_id="x",
        expected_citation_id="x",
        citation_match=citation_match,
        judge_score=judge_score,
        judge_bucket=judge_bucket,  # type: ignore[arg-type]
        judge_reasoning="r",
    )


class TestAggregateRows:
    def test_overall_metrics(self) -> None:
        rows = [
            _row("g1", citation_match=1, judge_score=1.0),
            _row("g2", citation_match=0, judge_score=0.5),
            _row("g3", citation_match=1, judge_score=0.8),
        ]
        agg = aggregate_rows(rows)
        assert agg.overall.n == 3
        assert agg.overall.tier1_citation_match == pytest.approx(2 / 3)
        assert agg.overall.tier2_judge_mean == pytest.approx((1.0 + 0.5 + 0.8) / 3)
        assert agg.overall.tier2_judge_pct_above_0_7 == pytest.approx(2 / 3)

    def test_per_target_jurisdiction(self) -> None:
        rows = [
            _row("g1", target_jurisdiction="sg", citation_match=1, judge_score=1.0),
            _row("g2", target_jurisdiction="sg", citation_match=0, judge_score=0.0),
            _row("g3", target_jurisdiction="th", citation_match=1, judge_score=0.9),
        ]
        agg = aggregate_rows(rows)
        assert agg.by_target_jurisdiction["sg"].tier1_citation_match == pytest.approx(0.5)
        assert agg.by_target_jurisdiction["th"].tier1_citation_match == pytest.approx(1.0)

    def test_per_target_language(self) -> None:
        rows = [
            _row("g1", target_language="en", judge_score=0.8),
            _row("g2", target_language="th", judge_score=0.3),
        ]
        agg = aggregate_rows(rows)
        assert agg.by_target_language["en"].tier2_judge_mean == pytest.approx(0.8)
        assert agg.by_target_language["th"].tier2_judge_mean == pytest.approx(0.3)

    def test_per_framework_pair(self) -> None:
        rows = [
            _row("g1", source_framework="gdpr", target_framework="pdpa_sg", citation_match=1),
            _row("g2", source_framework="gdpr", target_framework="pdpa_sg", citation_match=0),
            _row("g3", source_framework="gdpr", target_framework="pdpa_th", citation_match=1),
        ]
        agg = aggregate_rows(rows)
        assert agg.by_framework_pair["gdpr__pdpa_sg"].n == 2
        assert agg.by_framework_pair["gdpr__pdpa_sg"].tier1_citation_match == pytest.approx(0.5)
        assert agg.by_framework_pair["gdpr__pdpa_th"].tier1_citation_match == pytest.approx(1.0)

    def test_empty_rows_no_division_error(self) -> None:
        agg = aggregate_rows([])
        assert agg.overall.n == 0
        assert agg.overall.tier1_citation_match == 0.0


class TestBucketCounts:
    def test_counts_every_bucket(self) -> None:
        rows = [
            _row("g1", judge_bucket="exact"),
            _row("g2", judge_bucket="exact"),
            _row("g3", judge_bucket="wrong"),
        ]
        c = bucket_counts(rows)
        assert c["exact"] == 2
        assert c["wrong"] == 1
        assert c["partial_right"] == 0
        # All five buckets present in the result
        assert set(c.keys()) == {
            "wrong",
            "partial_wrong",
            "partial_right",
            "substantively_right",
            "exact",
        }
