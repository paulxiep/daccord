"""Build the retrieval-baseline FAISS index from a gold-split JSONL.

Tier-12B helper. Run once per train-split version; rerun when the train
split refreshes (the dataset SHA in the gold file changes). Writes two
artifacts under `data/indices/`:

    <output_stem>.faiss   — FAISS IndexFlatIP over L2-normalized MPNet
                            embeddings of every gold pair's source clause.
    <output_stem>.jsonl   — parallel RetrievalIndexEntry rows.

The same artifact pair is consumed by:
  - the retrieval baseline in `run_eval.py --models retrieval`
  - the serving-time hybrid router (`daccord.serving.hybrid.HybridRouter`)

So building once at tier 12B covers both consumers.

Example:

    cd envs/eval && uv run python scripts/build_retrieval_index.py \\
        --gold-path ../../data/splits/train.jsonl \\
        --output ../../data/indices/retrieval__train__<dataset_hash>

Embedder defaults to `paraphrase-multilingual-mpnet-base-v2` — multilingual
(EN/TH/FR/DE/Bahasa coverage), CPU-friendly, ~470MB model.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from daccord.eval.retrieval_index import build_index
from daccord.gold import GoldSet

log = logging.getLogger("build_retrieval_index")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EMBEDDER = "paraphrase-multilingual-mpnet-base-v2"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold-path",
        type=Path,
        required=True,
        help="path to gold JSONL (typically the train split — pairs in this file get indexed)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "output path stem (e.g. data/indices/retrieval__train__<hash>); "
            ".faiss + .jsonl suffixes appended. Existing files are overwritten."
        ),
    )
    parser.add_argument(
        "--embedder",
        default=DEFAULT_EMBEDDER,
        help=f"sentence-transformers model name (default: {DEFAULT_EMBEDDER})",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    gold = GoldSet.from_jsonl(args.gold_path)
    log.info(
        "loaded %d gold pairs from %s (dataset_hash=%s)",
        len(gold.pairs),
        gold.source_path,
        gold.dataset_hash,
    )
    log.info("embedding with %s (this downloads the model on first use)", args.embedder)

    faiss_path, jsonl_path = build_index(gold, args.embedder, args.output)
    log.info("wrote %s (%d vectors)", faiss_path, len(gold.pairs))
    log.info("wrote %s", jsonl_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
