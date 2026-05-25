"""Eval-harness runner.

Wires the schema + clients + scoring layer into one end-to-end call that
produces both the per-row CSV (the M0 deliverable) and an MLflow
parent-with-nested-children run shape for the per-model comparison.

The runner is the only place that knows about all four moving parts; the
layers below are pure (no MLflow, no CSV, no provider strings). That
keeps each module independently unit-testable and the runner integration-
testable on its own.

Default judge is `GeminiJudge` (free-tier). Caller may swap via the
`judge` arg — e.g., M4 may want a non-Gemini judge when a Gemini
generator is in the mix, to avoid self-judging bias.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import mlflow

from daccord.eval.clients import ModelClient
from daccord.eval.prompts import build_eval_prompt
from daccord.eval.scoring import (
    AggregateBreakdown,
    EvalAggregates,
    EvalRow,
    JudgeClient,
    aggregate_rows,
    bucket_counts,
    build_eval_row,
    judge_pair,
)
from daccord.gold import GoldSet
from daccord.tracking import (
    PROJECT_TAG_KEY,
    PROJECT_TAG_VALUE,
    log_standard_params,
    setup_mlflow,
)
from daccord.validation import ValidatedModel, validated

CSV_HEADER = (
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
EVAL_EXPERIMENT = "daccord-eval"


class EvalReport(ValidatedModel):
    """Final output of `run_eval`. Mirrors the artifacts written to disk +
    the MLflow run params/metrics, so callers can introspect a run
    without re-reading the CSV or hitting the tracking store.

    `slice_tag` records which stratified slice (`in-domain` /
    `out-of-domain` / `combined`) this run covered — set via the
    `--slice-tag` CLI flag at tier 12B. None for ungratified runs
    (M0 / toy gold).
    """

    run_name: str
    dataset_hash: str
    prompt_variant: str
    judge_model: str
    csv_path: str
    rows: list[EvalRow]
    per_model: dict[str, EvalAggregates]
    slice_tag: str | None = None


@validated
def write_csv(rows: list[EvalRow], path: Path) -> None:
    """Write per-row eval CSV. Column order is the wire contract; see
    [eval/README.md] for the M0-vs-M4 stability promise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(CSV_HEADER)
        writer.writerows(
            (
                r.gold_id,
                r.model,
                r.source_jurisdiction,
                r.source_framework,
                r.target_jurisdiction,
                r.target_framework,
                r.source_language,
                r.target_language,
                r.predicted_citation_id,
                r.expected_citation_id,
                r.citation_match,
                f"{r.judge_score:.4f}",
                r.judge_bucket,
                # Tabs and newlines in reasoning would break naive CSV
                # consumers. csv.writer handles quoting; we only strip
                # newlines so spreadsheet apps render one row per gold pair.
                r.judge_reasoning.replace("\n", " ").replace("\r", " "),
            )
            for r in rows
        )


def _log_breakdown(metric_prefix: str, breakdowns: dict[str, AggregateBreakdown]) -> None:
    for slice_key, b in breakdowns.items():
        # MLflow metric names allow alphanumerics, underscores, dots, slashes, hyphens.
        safe = slice_key.replace("/", "_")
        mlflow.log_metric(f"{metric_prefix}__{safe}", b.tier1_citation_match)


def _safe_name(s: str) -> str:
    """Normalize model strings to MLflow-safe run name fragments."""
    return s.replace("/", "_").replace(":", "_")


def _run_pair_major(
    generators: Sequence[ModelClient],
    judge: JudgeClient,
    gold: GoldSet,
    run_name: str,
) -> dict[str, list[EvalRow]]:
    """Pair-major execution: for each pair, call every generator (judging
    each response immediately), then move to the next pair.

    Versus the older generator-major order (all pairs through model A, then
    all through model B, …), pair-major:
      - **spreads each provider's calls across the run** so per-provider
        RPM ceilings (Gemini 15 RPM, Groq preview limits, etc.) see lower
        peak density than in the old contiguous batches;
      - lets the cooldown between consecutive same-provider calls grow
        naturally (every call cycles through N-1 other models first);
      - keeps the CSV row order generator-major for backward compatibility
        — the caller re-orders by iterating `generators` over the returned
        dict, so the wire contract (per [eval/README.md]) is unchanged.
    """
    per_model_rows: dict[str, list[EvalRow]] = {gen.model: [] for gen in generators}
    for pair in gold.pairs:
        prompt = build_eval_prompt(pair)
        for gen in generators:
            response = gen.generate(
                prompt, run_id=run_name, batch_id=f"{pair.framework_pair}::{pair.id}"
            )
            score = judge_pair(
                pair,
                response,
                judge,
                run_id=run_name,
                batch_id=f"{pair.framework_pair}::{pair.id}",
            )
            per_model_rows[gen.model].append(build_eval_row(pair, response, score))
    return per_model_rows


def run_eval(
    gold_path: Path,
    generators: Sequence[ModelClient],
    judge: JudgeClient,
    output_csv: Path,
    *,
    run_name: str,
    prompt_variant: str = "unconstrained-m0",
    seed: int = 42,
    slice_tag: str | None = None,
) -> EvalReport:
    """End-to-end eval run.

    Loads gold from JSONL, runs each generator over every pair, judges
    each result, writes the per-row CSV, and logs the MLflow parent +
    nested children with the metric schema described in
    [eval/README.md].

    `prompt_variant` is propagated to MLflow tags and is REQUIRED in the
    function signature (no silent default that drifts between M0 and M4).

    `slice_tag` (`in-domain` / `out-of-domain` / `combined` / None) tags
    this run as one slice of the tier-12B stratified eval. Set on both
    parent + nested child runs so MLflow filters can split them; intentionally
    *not* a per-row CSV column (slice is run-level metadata, the CSV row
    contract stays stable per [eval/README.md]).

    `gate` MLflow tag is set to `M4` when `slice_tag` is set (stratified
    eval is a tier-12B/M4 deliverable) and `M0` otherwise (untagged toy
    runs). This is a heuristic — callers wanting a different gate should
    set the tag manually via the MLflow client.
    """
    gold = GoldSet.from_jsonl(gold_path)
    setup_mlflow(experiment_name=EVAL_EXPERIMENT)

    all_rows: list[EvalRow] = []
    per_model: dict[str, EvalAggregates] = {}
    judge_model_name = judge.model
    gate_tag = "M4" if slice_tag is not None else "M0"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag(PROJECT_TAG_KEY, PROJECT_TAG_VALUE)
        mlflow.set_tag("gate", gate_tag)
        mlflow.set_tag("prompt_variant", prompt_variant)
        mlflow.set_tag("judge_model", judge_model_name)
        if slice_tag is not None:
            mlflow.set_tag("slice_tag", slice_tag)
        log_standard_params(
            run_name=run_name,
            seed=seed,
            dataset_hash=gold.dataset_hash,
            extra={
                "judge_model": judge_model_name,
                "prompt_variant": prompt_variant,
                "n_gold_pairs": str(len(gold.pairs)),
                "n_generators": str(len(generators)),
                **({"slice_tag": slice_tag} if slice_tag is not None else {}),
            },
        )

        # Pair-major execution — generates + judges every pair through every
        # generator in interleaved order. Returns per-model row lists; we
        # then log per-model MLflow children + aggregates after the loop.
        per_model_rows = _run_pair_major(generators, judge, gold, run_name)

        for gen in generators:
            rows = per_model_rows[gen.model]
            agg = aggregate_rows(rows)
            per_model[gen.model] = agg
            all_rows.extend(rows)  # generator-major CSV row order preserved

            child_name = f"{run_name}/{_safe_name(gen.model)}"
            with mlflow.start_run(run_name=child_name, nested=True):
                mlflow.set_tag(PROJECT_TAG_KEY, PROJECT_TAG_VALUE)
                mlflow.set_tag("prompt_variant", prompt_variant)
                mlflow.set_tag("judge_model", judge_model_name)
                if slice_tag is not None:
                    mlflow.set_tag("slice_tag", slice_tag)
                mlflow.log_params(
                    {
                        "model": gen.model,
                        "provider": gen.provider,
                        "judge_model": judge_model_name,
                        "dataset_hash": gold.dataset_hash,
                        "prompt_variant": prompt_variant,
                        "seed": seed,
                        **({"slice_tag": slice_tag} if slice_tag is not None else {}),
                    }
                )
                mlflow.log_metric("tier1_citation_match_overall", agg.overall.tier1_citation_match)
                mlflow.log_metric("tier2_judge_mean", agg.overall.tier2_judge_mean)
                mlflow.log_metric(
                    "tier2_judge_pct_above_0_7", agg.overall.tier2_judge_pct_above_0_7
                )
                mlflow.log_metric("n_pairs", float(agg.overall.n))
                _log_breakdown("tier1_citation_match__jur", agg.by_target_jurisdiction)
                _log_breakdown("tier1_citation_match__lang", agg.by_target_language)
                _log_breakdown("tier1_citation_match__fwpair", agg.by_framework_pair)
                for bucket, n in bucket_counts(rows).items():
                    mlflow.log_metric(f"judge_bucket_{bucket}", float(n))

        write_csv(all_rows, output_csv)
        mlflow.log_artifact(str(output_csv))

    return EvalReport(
        run_name=run_name,
        dataset_hash=gold.dataset_hash,
        prompt_variant=prompt_variant,
        judge_model=judge_model_name,
        csv_path=str(output_csv.as_posix()),
        rows=all_rows,
        per_model=per_model,
        slice_tag=slice_tag,
    )
