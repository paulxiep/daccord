"""End-to-end runner tests with mocked generator + judge.

Verifies the three things the runner is responsible for:
  1. Per-row CSV header + row contract is stable (regression guard for the
     wire format that future tools will parse).
  2. MLflow parent + nested children land in the `daccord-eval` experiment
     with the metric schema documented in the plan.
  3. Aggregates in the returned `EvalReport` agree with hand-computed
     values from the mocked generator outputs.

No live API calls. Uses the `isolated_tracking` fixture pattern from
[tests/test_tracking.py] to keep MLflow runs out of `./mlruns`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from daccord.costs.config import Provider
from daccord.eval.runner import CSV_HEADER, EVAL_EXPERIMENT, run_eval
from daccord.eval.schema import CitationCandidate, ModelResponse, PromptMessages
from daccord.eval.scoring import JudgeScore

TOY_ROWS = [
    {
        "id": "tg-001",
        "source_jurisdiction": "eu",
        "source_framework": "gdpr",
        "source_citation_id": "Art. 32",
        "source_mechanism": "Security of processing.",
        "source_language": "en",
        "target_jurisdiction": "sg",
        "target_framework": "pdpa_sg",
        "target_citation_id": "Section 24",
        "target_mechanism": "Reasonable security arrangements.",
        "target_language": "en",
        "notes": None,
    },
    {
        "id": "tg-002",
        "source_jurisdiction": "eu",
        "source_framework": "gdpr",
        "source_citation_id": "Art. 17",
        "source_mechanism": "Right to erasure.",
        "source_language": "en",
        "target_jurisdiction": "th",
        "target_framework": "pdpa_th",
        "target_citation_id": "Section 33",
        "target_mechanism": "Right to be forgotten under Thai PDPA.",
        "target_language": "th",
        "notes": None,
    },
    {
        "id": "tg-003",
        "source_jurisdiction": "eu",
        "source_framework": "gdpr",
        "source_citation_id": "Art. 5",
        "source_mechanism": "Lawfulness, fairness, transparency.",
        "source_language": "en",
        "target_jurisdiction": "sg",
        "target_framework": "pdpa_sg",
        "target_citation_id": "Section 13",
        "target_mechanism": "Consent obligation.",
        "target_language": "en",
        "notes": None,
    },
]


@pytest.fixture
def isolated_tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    uri = f"file:{(tmp_path / 'mlruns').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return tmp_path


@pytest.fixture
def gold_file(tmp_path: Path) -> Path:
    p = tmp_path / "toy.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in TOY_ROWS) + "\n", encoding="utf-8")
    return p


class _FakeGenerator:
    """Returns a hand-picked citation per gold_id so we can compute expected
    aggregates without invoking a real model.
    """

    provider: Provider = "groq"

    def __init__(self, model: str, citation_map: dict[str, str | None]) -> None:
        self.model = model
        self._map = citation_map

    def generate(self, messages: PromptMessages, *, run_id: str, batch_id: str) -> ModelResponse:
        # Identify gold by batch_id suffix (set by runner as fwpair::id)
        gold_id = batch_id.rsplit("::", 1)[-1]
        citation = self._map.get(gold_id)
        if citation is None:
            return ModelResponse(
                model=self.model,
                top1=None,
                raw_text="",
                input_tokens=10,
                output_tokens=0,
                latency_ms=1.0,
                parse_error="forced parse failure",
            )
        return ModelResponse(
            model=self.model,
            top1=CitationCandidate(
                citation_id=citation,
                target_mechanism="m",
                mapping_justification="j",
            ),
            raw_text=json.dumps({"citation_id": citation}),
            input_tokens=10,
            output_tokens=20,
            latency_ms=1.0,
        )


class _FakeJudge:
    provider: Provider = "google_gemini"

    def __init__(self, model: str = "fake-judge") -> None:
        self.model = model

    def judge(self, messages: PromptMessages, *, run_id: str, batch_id: str) -> JudgeScore:
        # Score = 1.0 if "Section 24" or "Section 33" or "Section 13" appears
        # (the gold answers) in the rendered prompt; 0.2 otherwise.
        text = messages.user
        if any(g in text for g in ("Section 24", "Section 33", "Section 13")):
            return JudgeScore(score=1.0, bucket="exact", reasoning="ok", judge_model=self.model)
        return JudgeScore(
            score=0.2, bucket="partial_wrong", reasoning="off", judge_model=self.model
        )


class TestRunEval:
    def test_csv_header_is_stable(self) -> None:
        # Wire contract regression guard
        assert CSV_HEADER == (
            "gold_id",
            "model",
            "source_jurisdiction",
            "source_framework",
            "target_jurisdiction",
            "target_framework",
            "source_language",
            "target_language",
            "predicted_citation_id",
            "expected_citation_id",
            "citation_match",
            "judge_score",
            "judge_bucket",
            "judge_reasoning",
        )

    def test_end_to_end_one_generator(
        self, gold_file: Path, isolated_tracking: Path, tmp_path: Path
    ) -> None:
        # tg-001 right, tg-002 wrong (citation but mismatched), tg-003 parse failure
        gen = _FakeGenerator(
            "fake-llama",
            {"tg-001": "Section 24", "tg-002": "Section 99", "tg-003": None},
        )
        out_csv = tmp_path / "out.csv"
        report = run_eval(
            gold_path=gold_file,
            generators=[gen],
            judge=_FakeJudge(),
            output_csv=out_csv,
            run_name="test-run-1",
        )

        assert report.run_name == "test-run-1"
        assert report.judge_model == "fake-judge"
        assert report.csv_path == out_csv.as_posix()
        assert len(report.rows) == 3
        assert "fake-llama" in report.per_model

        agg = report.per_model["fake-llama"].overall
        # Tier 1: 1 of 3 matched (tg-001 only — candidate citation == gold)
        assert agg.tier1_citation_match == pytest.approx(1 / 3)
        # Judge fakes: 1.0 whenever a canonical gold-answer string appears in
        # the prompt; the prompt embeds BOTH gold and candidate, so tg-001
        # and tg-002 both see "Section 24"/"Section 33" in the gold line.
        # tg-003 short-circuits to 0.0 (parse failure → judge not called).
        # The end-to-end plumbing is what's under test; judge discrimination
        # is a separate concern owned by the real judge.
        assert agg.tier2_judge_mean == pytest.approx((1.0 + 1.0 + 0.0) / 3)

    def test_csv_contents_match_header_and_rows(
        self, gold_file: Path, isolated_tracking: Path, tmp_path: Path
    ) -> None:
        gen = _FakeGenerator(
            "fake-m",
            {"tg-001": "Section 24", "tg-002": "Sec. 33", "tg-003": "Section 13"},
        )
        out_csv = tmp_path / "out.csv"
        run_eval(
            gold_path=gold_file,
            generators=[gen],
            judge=_FakeJudge(),
            output_csv=out_csv,
            run_name="test-run-2",
        )
        with out_csv.open(encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            rows = list(reader)
        assert rows[0] == list(CSV_HEADER)
        assert len(rows) == 4  # header + 3 data
        # Every data row's citation_match should be "1" (all gold-matched after
        # normalize: "Sec. 33" -> "33", "Section 33" -> "33")
        for r in rows[1:]:
            assert r[CSV_HEADER.index("citation_match")] == "1"

    def test_two_generators_produce_distinct_per_model_aggregates(
        self, gold_file: Path, isolated_tracking: Path, tmp_path: Path
    ) -> None:
        # gen_a gets everything right; gen_b gets nothing right.
        gen_a = _FakeGenerator(
            "fake-a", {"tg-001": "Section 24", "tg-002": "Section 33", "tg-003": "Section 13"}
        )
        gen_b = _FakeGenerator(
            "fake-b", {"tg-001": "Section 99", "tg-002": "Section 99", "tg-003": "Section 99"}
        )
        out_csv = tmp_path / "out.csv"
        report = run_eval(
            gold_path=gold_file,
            generators=[gen_a, gen_b],
            judge=_FakeJudge(),
            output_csv=out_csv,
            run_name="two-gens",
        )
        assert report.per_model["fake-a"].overall.tier1_citation_match == pytest.approx(1.0)
        assert report.per_model["fake-b"].overall.tier1_citation_match == pytest.approx(0.0)
        # CSV is the union of both models' rows
        assert len(report.rows) == 6

    def test_mlflow_parent_and_nested_children_logged(
        self, gold_file: Path, isolated_tracking: Path, tmp_path: Path
    ) -> None:
        gen = _FakeGenerator(
            "fake-m",
            {"tg-001": "Section 24", "tg-002": "Section 33", "tg-003": "Section 13"},
        )
        out_csv = tmp_path / "out.csv"
        run_eval(
            gold_path=gold_file,
            generators=[gen],
            judge=_FakeJudge(),
            output_csv=out_csv,
            run_name="mlflow-shape-test",
        )

        client = MlflowClient()
        exp = client.get_experiment_by_name(EVAL_EXPERIMENT)
        assert exp is not None
        runs = client.search_runs([exp.experiment_id])
        # 1 parent + 1 nested child
        assert len(runs) == 2
        # Identify by run_name tag (mlflow stores it under tags.mlflow.runName)
        parents = [r for r in runs if r.data.tags.get("mlflow.runName") == "mlflow-shape-test"]
        children = [
            r
            for r in runs
            if r.data.tags.get("mlflow.runName", "").startswith("mlflow-shape-test/")
        ]
        assert len(parents) == 1
        assert len(children) == 1

        parent = parents[0]
        # Parent tags
        assert parent.data.tags["project"] == "daccord"
        assert parent.data.tags["gate"] == "M0"
        assert parent.data.tags["prompt_variant"] == "unconstrained-m0"
        # Parent params
        assert parent.data.params["judge_model"] == "fake-judge"
        assert parent.data.params["n_gold_pairs"] == "3"
        assert parent.data.params["n_generators"] == "1"
        assert "dataset_hash" in parent.data.params

        child = children[0]
        assert child.data.params["model"] == "fake-m"
        assert child.data.params["provider"] == "groq"
        # Child metrics — must include the headline tier-1/tier-2 numbers
        assert "tier1_citation_match_overall" in child.data.metrics
        assert "tier2_judge_mean" in child.data.metrics
        assert "tier2_judge_pct_above_0_7" in child.data.metrics
        # Per-jurisdiction breakdown — both sg and th from TOY_ROWS appear
        assert "tier1_citation_match__jur__sg" in child.data.metrics
        assert "tier1_citation_match__jur__th" in child.data.metrics
        # Per-language: en + th
        assert "tier1_citation_match__lang__en" in child.data.metrics
        assert "tier1_citation_match__lang__th" in child.data.metrics
        # Per-framework-pair
        assert "tier1_citation_match__fwpair__gdpr__pdpa_sg" in child.data.metrics
        assert "tier1_citation_match__fwpair__gdpr__pdpa_th" in child.data.metrics
        # Bucket histogram
        for b in ("wrong", "partial_wrong", "partial_right", "substantively_right", "exact"):
            assert f"judge_bucket_{b}" in child.data.metrics

    def test_reasoning_newlines_stripped_in_csv(
        self, gold_file: Path, isolated_tracking: Path, tmp_path: Path
    ) -> None:
        class _NewlineJudge:
            provider: Provider = "google_gemini"
            model = "n-j"

            def judge(self, messages: PromptMessages, *, run_id: str, batch_id: str) -> JudgeScore:
                return JudgeScore(
                    score=0.5,
                    bucket="partial_right",
                    reasoning="line one\nline two\r\nline three",
                    judge_model=self.model,
                )

        gen = _FakeGenerator(
            "fake-m", {"tg-001": "Section 24", "tg-002": "Section 33", "tg-003": "Section 13"}
        )
        out_csv = tmp_path / "out.csv"
        run_eval(
            gold_path=gold_file,
            generators=[gen],
            judge=_NewlineJudge(),
            output_csv=out_csv,
            run_name="newline-test",
        )
        with out_csv.open(encoding="utf-8") as fh:
            text = fh.read()
        # 1 header + 3 rows = 4 lines. Newlines inside reasoning would inflate.
        assert text.count("\n") == 4
