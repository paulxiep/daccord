"""CLI wrapper for the eval harness (tier 2B + 3A).

Examples (run via Docker Compose — see CLAUDE.md):

    # Smoke against the toy gold with Llama 4 (Groq) + Gemini 3.1 Flash Lite,
    # judged by Llama 4 Scout via Groq:
    docker compose run --rm eval uv run python scripts/run_eval.py \\
        --gold-path ../../data/gold/toy_v1.jsonl \\
        --models groq,gemini \\
        --judge meta-llama/llama-4-scout-17b-16e-instruct \\
        --output-csv ../../eval/baseline_toy.csv \\
        --run-name baseline-toy-2026-05-25

    # Add the local Qwen3-8B 4-bit-NF4 baseline + Qwen3-32B-via-Groq (tier 3A);
    # runs in the `baseline` GPU service so torch + bitsandbytes are available:
    docker compose run --rm baseline uv run python ../eval/scripts/run_eval.py \\
        --gold-path ../../data/gold/toy_v1.jsonl \\
        --models qwen,groq,qwen3,gemini \\
        --judge meta-llama/llama-4-scout-17b-16e-instruct \\
        --output-csv ../../eval/baseline_toy.csv \\
        --run-name baseline-toy-2026-05-25

    # Dry run — validates schema + builds prompts, makes no API calls
    docker compose run --rm eval uv run python scripts/run_eval.py \\
        --gold-path ../../data/gold/toy_v1.jsonl --dry-run

Defaults are intentionally minimal: model strings are short aliases mapped to
the concrete `(provider, model)` pairs. Pass `--groq-model <id>`,
`--gemini-model <id>`, or `--qwen-model <id>` to override.

Requires GROQ_API_KEY and/or GOOGLE_API_KEY in the environment — load via
`.env.local` (caller exports before invoking). The harness does NOT pull
in python-dotenv to keep the dep tree lean. The `qwen` alias has no API
key requirement (local inference) but expects a CUDA-capable GPU.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from daccord.eval.clients import (
    GeminiClient,
    GroqClient,
    LocalHFClient,
    ModelClient,
    RetrievalClient,
)
from daccord.eval.runner import run_eval
from daccord.eval.scoring import GeminiJudge, GroqJudge, JudgeClient

log = logging.getLogger("run_eval")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GOLD = REPO_ROOT / "data" / "gold" / "toy_v1.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "eval" / "baseline_toy.csv"

MODEL_ALIASES: dict[str, str] = {
    # API generators — current-generation free-tier comparators (2026).
    "groq": "meta-llama/llama-4-scout-17b-16e-instruct",  # Llama 4 Scout via Groq
    "qwen3": "qwen/qwen3-32b",  # frontier Qwen via Groq (separate from local Qwen base)
    "gemini": "gemini-3.1-flash-lite",
    # Local — QLoRA base for the fine-tune (RTX 5080 16 GB, 4-bit NF4). Chosen
    # after the 2026-05-25 base-revisit: Qwen3-8B replaces Qwen2.5-7B-Instruct
    # (the latter was picked by the original tokenizer audit but Qwen 3 has a
    # newer multilingual tokenizer and similar VRAM footprint).
    "qwen": "Qwen/Qwen3-8B",
    # Retrieval baseline (tier 12B).
    "retrieval": "retrieval/paraphrase-multilingual-mpnet-base-v2",
}
DEFAULT_RETRIEVAL_EMBEDDER = "paraphrase-multilingual-mpnet-base-v2"


def _resolve_generators(
    aliases: list[str],
    groq_model: str | None,
    qwen3_model: str | None,
    gemini_model: str | None,
    qwen_model: str | None,
    retrieval_index_path: Path | None,
    retrieval_embedder: str,
    retrieval_score_threshold: float | None,
) -> list[ModelClient]:
    out: list[ModelClient] = []
    for a in aliases:
        if a == "groq":
            out.append(GroqClient(model=groq_model or MODEL_ALIASES["groq"]))
        elif a == "qwen3":
            # Qwen 3-32B via Groq — different model_id but same SDK as GroqClient.
            out.append(GroqClient(model=qwen3_model or MODEL_ALIASES["qwen3"]))
        elif a == "gemini":
            out.append(GeminiClient(model=gemini_model or MODEL_ALIASES["gemini"]))
        elif a == "qwen":
            out.append(LocalHFClient(model=qwen_model or MODEL_ALIASES["qwen"]))
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
    # Default judge is Llama 4 Scout via Groq (strongest current-generation
    # free-tier Llama; bumped 2026-05-25 from llama-3.3-70b-versatile). Gemini
    # judge kept as alternative for when M4 swaps Llama into the generator
    # pool and wants a different judge family.
    groq_aliases = (
        "groq",
        "llama-4-scout",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "llama-3.3-70b-versatile",
    )
    if (
        judge_arg in groq_aliases
        or judge_arg.startswith("llama-")
        or judge_arg.startswith("meta-llama/")
    ):
        if judge_arg.startswith("llama-") or judge_arg.startswith("meta-llama/"):
            model = judge_arg
        else:
            model = "meta-llama/llama-4-scout-17b-16e-instruct"
        return GroqJudge(model=model)
    if judge_arg in ("gemini", "gemini-3.1-flash-lite") or judge_arg.startswith("gemini-"):
        model = judge_arg if judge_arg.startswith("gemini-") else "gemini-3.1-flash-lite"
        return GeminiJudge(model=model)
    raise SystemExit(
        f"unknown judge {judge_arg!r}; supported: groq, llama-3.3-70b-versatile, "
        "gemini, gemini-3.1-flash-lite"
    )


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
        log.info(
            "[dry-run] sample prompt (first pair):\n--- system ---\n%s\n--- user ---\n%s",
            sample.system,
            sample.user,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=DEFAULT_GOLD,
        help=f"path to gold JSONL (default: {DEFAULT_GOLD})",
    )
    parser.add_argument(
        "--models",
        default="groq,gemini",
        help="comma-separated model aliases: groq,qwen3,gemini,qwen,retrieval",
    )
    parser.add_argument(
        "--judge",
        default="meta-llama/llama-4-scout-17b-16e-instruct",
        help=(
            "judge model (default: llama-4-scout via Groq). Also supports: "
            "groq, llama-3.3-70b-versatile, gemini, gemini-3.1-flash-lite."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"per-row CSV output (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--run-name",
        required=False,
        default=None,
        help="MLflow parent run name (required for live runs)",
    )
    parser.add_argument(
        "--prompt-variant",
        default="unconstrained-m0",
        help="propagated to MLflow tags + downstream comparability notes",
    )
    parser.add_argument(
        "--groq-model",
        default=None,
        help=("override Groq alias model id (default: meta-llama/llama-4-scout-17b-16e-instruct)"),
    )
    parser.add_argument(
        "--qwen3-model",
        default=None,
        help="override Qwen3 alias model id (default: qwen/qwen3-32b via Groq)",
    )
    parser.add_argument(
        "--gemini-model",
        default=None,
        help="override Gemini model id (default: gemini-3.1-flash-lite)",
    )
    parser.add_argument(
        "--qwen-model",
        default=None,
        help="override local Qwen HF id (default: Qwen/Qwen3-8B, 4-bit NF4)",
    )
    parser.add_argument(
        "--retrieval-index-path",
        type=Path,
        default=None,
        help="path to FAISS index (required when retrieval is in --models)",
    )
    parser.add_argument(
        "--retrieval-embedder",
        default=DEFAULT_RETRIEVAL_EMBEDDER,
        help=f"retrieval embedder (default: {DEFAULT_RETRIEVAL_EMBEDDER})",
    )
    parser.add_argument(
        "--retrieval-score-threshold",
        type=float,
        default=None,
        help=(
            "optional cosine ceiling; top-1 below this returns top1=None "
            "(Tier-1 miss). Default: no threshold (always return top-1)."
        ),
    )
    parser.add_argument(
        "--slice-tag",
        default=None,
        choices=["in-domain", "out-of-domain", "combined"],
        help=(
            "tag this run as in-domain / out-of-domain / combined; "
            "written to MLflow run tag (not CSV row). Default: untagged."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run", action="store_true", help="validate gold + build prompts, no API calls"
    )
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
        args.qwen3_model,
        args.gemini_model,
        args.qwen_model,
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
    log.info(
        "wrote %s (%d rows across %d models)",
        report.csv_path,
        len(report.rows),
        len(report.per_model),
    )
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
