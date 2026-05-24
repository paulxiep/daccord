"""CLI wrapper for the eval harness (tier 2B).

Examples:

    # Smoke against the toy gold with Groq + Gemini, judged by Gemini Flash
    uv run python eval/run_eval.py \\
        --gold-path data/gold/toy_v1.jsonl \\
        --models groq,gemini \\
        --judge gemini-2.5-flash \\
        --output-csv eval/baseline_toy.csv \\
        --run-name baseline-toy-2026-05-23

    # Dry run — validates schema + builds prompts, makes no API calls
    uv run python eval/run_eval.py --gold-path data/gold/toy_v1.jsonl --dry-run

Defaults are intentionally minimal: model strings are short aliases mapped to
the concrete `(provider, model)` pairs. Pass `--groq-model <id>` or
`--gemini-model <id>` to override.

Requires GROQ_API_KEY and/or GOOGLE_API_KEY in the environment — load via
`.env.local` (caller exports before invoking). The harness does NOT pull
in python-dotenv to keep the dep tree lean.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from daccord.eval.clients import GeminiClient, GroqClient, ModelClient, RetrievalClient
from daccord.eval.runner import run_eval
from daccord.eval.scoring import GeminiJudge, JudgeClient

log = logging.getLogger("run_eval")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GOLD = REPO_ROOT / "data" / "gold" / "toy_v1.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "eval" / "baseline_toy.csv"

MODEL_ALIASES: dict[str, str] = {
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.5-flash",
    "retrieval": "retrieval/paraphrase-multilingual-mpnet-base-v2",
}
DEFAULT_RETRIEVAL_EMBEDDER = "paraphrase-multilingual-mpnet-base-v2"


def _resolve_generators(
    aliases: list[str],
    groq_model: str | None,
    gemini_model: str | None,
    retrieval_index_path: Path | None,
    retrieval_embedder: str,
    retrieval_score_threshold: float | None,
) -> list[ModelClient]:
    out: list[ModelClient] = []
    for a in aliases:
        if a == "groq":
            out.append(GroqClient(model=groq_model or MODEL_ALIASES["groq"]))
        elif a == "gemini":
            out.append(GeminiClient(model=gemini_model or MODEL_ALIASES["gemini"]))
        elif a == "retrieval":
            if retrieval_index_path is None:
                raise SystemExit(
                    "retrieval alias requires --retrieval-index-path (build via "
                    "scripts/build_retrieval_index.py)"
                )
            out.append(
                RetrievalClient(
                    index_path=retrieval_index_path,
                    embedder_name=retrieval_embedder,
                    score_threshold=retrieval_score_threshold,
                )
            )
        else:
            raise SystemExit(
                f"unknown model alias {a!r}; valid: {', '.join(sorted(MODEL_ALIASES))}"
            )
    return out


def _resolve_judge(judge_arg: str) -> JudgeClient:
    # M0: only Gemini Flash is wired as a judge. Other judges can be added
    # in this dispatch when needed (M4 may want a non-Gemini judge to
    # avoid self-judging when Gemini is also a generator).
    if judge_arg in ("gemini", "gemini-2.5-flash") or judge_arg.startswith("gemini-"):
        model = judge_arg if judge_arg.startswith("gemini-") else "gemini-2.5-flash"
        return GeminiJudge(model=model)
    raise SystemExit(f"unknown judge {judge_arg!r}; supported: gemini, gemini-2.5-flash")


def _dry_run(gold_path: Path, aliases: list[str]) -> int:
    """Schema + prompt validation only. Zero API calls, zero MLflow writes."""
    from daccord.eval.prompts import build_eval_prompt
    from daccord.gold import GoldSet

    gold = GoldSet.from_jsonl(gold_path)
    log.info("[dry-run] loaded %d pairs from %s", len(gold.pairs), gold.source_path)
    log.info("[dry-run] dataset_hash = %s", gold.dataset_hash)
    for alias in aliases:
        log.info(
            "[dry-run] would invoke alias=%s on model=%s",
            alias,
            MODEL_ALIASES.get(alias, "<unknown>"),
        )
    if gold.pairs:
        sample = build_eval_prompt(gold.pairs[0])
        log.info("[dry-run] sample prompt (first pair):\n--- system ---\n%s\n--- user ---\n%s",
                 sample.system, sample.user)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-path", type=Path, default=DEFAULT_GOLD,
                        help=f"path to gold JSONL (default: {DEFAULT_GOLD})")
    parser.add_argument("--models", default="groq,gemini",
                        help="comma-separated model aliases: groq,gemini,retrieval")
    parser.add_argument("--judge", default="gemini-2.5-flash",
                        help="judge model (default: gemini-2.5-flash)")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT,
                        help=f"per-row CSV output (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--run-name", required=False, default=None,
                        help="MLflow parent run name (required for live runs)")
    parser.add_argument("--prompt-variant", default="unconstrained-m0",
                        help="propagated to MLflow tags + downstream comparability notes")
    parser.add_argument("--groq-model", default=None,
                        help="override Groq model id (default: llama-3.3-70b-versatile)")
    parser.add_argument("--gemini-model", default=None,
                        help="override Gemini model id (default: gemini-2.5-flash)")
    parser.add_argument("--retrieval-index-path", type=Path, default=None,
                        help="path to FAISS index (required when retrieval is in --models)")
    parser.add_argument("--retrieval-embedder", default=DEFAULT_RETRIEVAL_EMBEDDER,
                        help=f"retrieval embedder (default: {DEFAULT_RETRIEVAL_EMBEDDER})")
    parser.add_argument("--retrieval-score-threshold", type=float, default=None,
                        help=(
                            "optional cosine ceiling; top-1 below this returns top1=None "
                            "(Tier-1 miss). Default: no threshold (always return top-1)."
                        ))
    parser.add_argument("--slice-tag", default=None,
                        choices=["in-domain", "out-of-domain", "combined"],
                        help=(
                            "tag this run as in-domain / out-of-domain / combined; "
                            "written to MLflow run tag (not CSV row). Default: untagged."
                        ))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="validate gold + build prompts, no API calls")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    aliases = [a.strip() for a in args.models.split(",") if a.strip()]
    if args.dry_run:
        return _dry_run(args.gold_path, aliases)

    if not args.run_name:
        log.error("--run-name is required for live runs (use e.g. baseline-toy-YYYY-MM-DD)")
        return 2

    generators = _resolve_generators(
        aliases,
        args.groq_model,
        args.gemini_model,
        args.retrieval_index_path,
        args.retrieval_embedder,
        args.retrieval_score_threshold,
    )
    judge = _resolve_judge(args.judge)
    report = run_eval(
        gold_path=args.gold_path,
        generators=generators,
        judge=judge,
        output_csv=args.output_csv,
        run_name=args.run_name,
        prompt_variant=args.prompt_variant,
        seed=args.seed,
        slice_tag=args.slice_tag,
    )
    log.info("wrote %s (%d rows across %d models)",
             report.csv_path, len(report.rows), len(report.per_model))
    for model, agg in report.per_model.items():
        log.info(
            "  %s: tier1=%.3f  tier2_mean=%.3f  pct>=0.7=%.3f  n=%d",
            model,
            agg.overall.tier1_citation_match,
            agg.overall.tier2_judge_mean,
            agg.overall.tier2_judge_pct_above_0_7,
            agg.overall.n,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
