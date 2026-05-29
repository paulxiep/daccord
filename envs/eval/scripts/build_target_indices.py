"""Build per-framework target-clause FAISS indices (tier-6B++ RAG seat input).

For each `data/clauses/{framework}.json` (tier-5 output), embeds every
`(citation_id, clause_text)` pair and writes a 2-artifact index pair under
`data/indices/target_clauses/`:

    {framework}.faiss   — FAISS IndexFlatIP over L2-normalised embeddings
    {framework}.jsonl   — parallel TargetClauseIndexEntry rows

The RAG seat (tier 6B++, at [src/daccord/ensemble/strategies/rag_seat.py])
queries these indices with source-clause text from any framework to find
the most semantically similar target citation — that becomes the seat's
5th "vote" in the tier-7A ensemble.

Idempotent: skip a framework when its `.faiss` + `.jsonl` are newer than
the input `data/clauses/{framework}.json` (avoid re-embedding ~9 frameworks
on every run; force re-build via `--no-skip-existing`).

Empty-text clauses (the ~5% `body_recall` gap from tier 5) are skipped —
indexing an empty embedding would be noise in the similarity search.

Embedder defaults to `paraphrase-multilingual-mpnet-base-v2` — same as the
tier-12B retrieval baseline; passed M0 tokenizer audit for th/fr/de/en.

Example:

    cd envs/eval && uv run python scripts/build_target_indices.py
    # or restrict:
    cd envs/eval && uv run python scripts/build_target_indices.py --frameworks gdpr pdpa_sg
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from daccord.eval.retrieval_index import build_target_clause_index

log = logging.getLogger("build_target_indices")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CLAUSES_DIR = REPO_ROOT / "data" / "clauses"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "indices" / "target_clauses"
DEFAULT_EMBEDDER = "paraphrase-multilingual-mpnet-base-v2"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clauses-dir",
        type=Path,
        default=DEFAULT_CLAUSES_DIR,
        help=(
            f"Directory with {{framework}}.json files (tier-5 output). "
            f"Default: {DEFAULT_CLAUSES_DIR}"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for per-framework indices. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--embedder",
        default=DEFAULT_EMBEDDER,
        help=f"sentence-transformers model name (default: {DEFAULT_EMBEDDER})",
    )
    parser.add_argument(
        "--frameworks",
        nargs="*",
        default=None,
        help="Restrict to specific framework IDs (e.g. gdpr pdpa_sg). Default: all.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-build all indices even when output is newer than input.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    clause_files = _discover_clause_files(args.clauses_dir, args.frameworks)
    if not clause_files:
        log.error("No clause files at %s/*.json", args.clauses_dir)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    built, skipped, skipped_empty_total = 0, 0, 0
    for clause_path in clause_files:
        framework = clause_path.stem
        out_stem = args.out_dir / framework

        if not args.no_skip_existing and _output_fresh(clause_path, out_stem):
            log.info("  %-15s skipped (output up-to-date)", framework)
            skipped += 1
            continue

        payload = json.loads(clause_path.read_text(encoding="utf-8"))
        clauses = payload.get("clauses") or {}
        if not isinstance(clauses, dict):
            log.error("%s: clauses field is %s, expected dict", clause_path, type(clauses).__name__)
            return 1

        non_empty = sum(1 for v in clauses.values() if v and v.strip())
        skipped_empty = len(clauses) - non_empty
        skipped_empty_total += skipped_empty

        faiss_path, jsonl_path = build_target_clause_index(
            framework=framework,
            clauses=clauses,
            embedder_name=args.embedder,
            output_path=out_stem,
        )
        log.info(
            "  %-15s built (%d clauses indexed, %d empty skipped) → %s",
            framework,
            non_empty,
            skipped_empty,
            faiss_path.name,
        )
        built += 1

    log.info(
        "Done: %d frameworks built, %d skipped (output fresh), %d empty clauses total",
        built,
        skipped,
        skipped_empty_total,
    )
    return 0


def _discover_clause_files(clauses_dir: Path, restrict_to: list[str] | None) -> list[Path]:
    all_files = sorted(clauses_dir.glob("*.json"))
    if restrict_to is None:
        return all_files
    wanted = set(restrict_to)
    return [p for p in all_files if p.stem in wanted]


def _output_fresh(input_path: Path, out_stem: Path) -> bool:
    """True if both index artifacts exist AND are newer than the clauses file."""
    faiss_path = out_stem.with_suffix(".faiss")
    jsonl_path = out_stem.with_suffix(".jsonl")
    if not faiss_path.exists() or not jsonl_path.exists():
        return False
    input_mtime = input_path.stat().st_mtime
    return faiss_path.stat().st_mtime >= input_mtime and jsonl_path.stat().st_mtime >= input_mtime


if __name__ == "__main__":
    sys.exit(main())
