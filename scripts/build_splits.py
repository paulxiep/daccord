"""Tier-7B CLI: build jurisdiction-disjoint train/val/test splits.

Reads `data/ensemble/tiered/*.jsonl` (tier-6B output) and optionally:
  - `data/ensemble/validated/*.jsonl` (tier-7C labeler verdicts)
  - `data/ensemble/bidirectional/*.jsonl` (tier-6B+ cross-check)

Writes:
  - `data/splits/{train,val,test}.jsonl` — one `TieredPair` per row
  - `data/splits/splits_manifest.json` — counts + jurisdictions + SHA256
    of every input tree

Defaults: test={th, ph}, val={my}, train=remaining. Tier floor = HIGH.
With `--promote-bidirectional-consistent`, MED rows confirmed by the
reverse-direction check go to gold without needing a reviewer verdict.

See [src/daccord/ensemble/splits.py](../src/daccord/ensemble/splits.py)
for the gold-eligibility rules and manifest schema.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import get_args

from daccord.ensemble.schema import Tier
from daccord.ensemble.splits import (
    DEFAULT_TEST_JURISDICTIONS,
    DEFAULT_VAL_JURISDICTIONS,
    build_splits,
)

log = logging.getLogger("build_splits")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    val_js = _csv(args.val_jurisdictions) or list(DEFAULT_VAL_JURISDICTIONS)
    test_js = _csv(args.test_jurisdictions) or list(DEFAULT_TEST_JURISDICTIONS)

    manifest = build_splits(
        tiered_dir=args.tiered_dir,
        out_dir=args.out_dir,
        validated_dir=args.validated_dir if args.validated_dir.exists() else None,
        bidirectional_dir=(args.bidirectional_dir if args.bidirectional_dir.exists() else None),
        promote_bidirectional_consistent=args.promote_bidirectional_consistent,
        promote_rag_concurs=args.promote_rag_concurs,
        tier_floor=args.tier_floor,
        val_jurisdictions=val_js,
        test_jurisdictions=test_js,
        write=not args.dry_run,
    )

    log.info(
        "tier_floor=%s  validated=%s  bidirectional=%s  promote_bi=%s  promote_rag=%s",
        manifest.tier_floor,
        "present" if manifest.validated_input_sha256 else "absent",
        "present" if manifest.bidirectional_input_sha256 else "absent",
        manifest.promote_bidirectional_consistent,
        manifest.promote_rag_concurs,
    )
    log.info(
        "  train: %4d rows  jurisdictions=%s",
        manifest.train.count,
        manifest.train.source_jurisdictions,
    )
    log.info(
        "  val:   %4d rows  jurisdictions=%s",
        manifest.val.count,
        manifest.val.source_jurisdictions,
    )
    log.info(
        "  test:  %4d rows  jurisdictions=%s",
        manifest.test.count,
        manifest.test.source_jurisdictions,
    )
    if args.dry_run:
        print(json.dumps(manifest.model_dump(), indent=2, ensure_ascii=False))
    else:
        log.info("Wrote %s/{train,val,test}.jsonl + splits_manifest.json", args.out_dir)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiered-dir",
        type=Path,
        default=Path("data/ensemble/tiered"),
        help="Directory of tier-6B output (*.jsonl).",
    )
    parser.add_argument(
        "--validated-dir",
        type=Path,
        default=Path("data/ensemble/validated"),
        help="Directory of tier-7C overlay (*.jsonl). Absent/empty = HIGH-only gold.",
    )
    parser.add_argument(
        "--bidirectional-dir",
        type=Path,
        default=Path("data/ensemble/bidirectional"),
        help="Directory of tier-6B+ bidirectional cross-check (*.jsonl). "
        "Used only when --promote-bidirectional-consistent is set.",
    )
    parser.add_argument(
        "--promote-bidirectional-consistent",
        action="store_true",
        help="Auto-promote MED rows with bidirectional status=consistent into gold "
        "without requiring a tier-7C reviewer verdict.",
    )
    parser.add_argument(
        "--promote-rag-concurs",
        action="store_true",
        help="Auto-promote MED rows where the tier-6B++ RAG seat's vote "
        "concurs with the LLM consensus (rag_concurs=True on the tiered row). "
        "Independent retrieval signal — distinct from bidirectional, opt-in.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/splits"),
        help="Output directory for {train,val,test}.jsonl + splits_manifest.json.",
    )
    parser.add_argument(
        "--tier-floor",
        type=str,
        choices=list(get_args(Tier)),
        default="HIGH",
        help="Minimum tier accepted for gold (HIGH = HIGH-only; MED/LOW also "
        "require an overlay-confirmed row).",
    )
    parser.add_argument(
        "--val-jurisdictions",
        type=str,
        default=None,
        help=f"Comma-list of source_jurisdiction codes for val split (default: "
        f"{','.join(DEFAULT_VAL_JURISDICTIONS)}).",
    )
    parser.add_argument(
        "--test-jurisdictions",
        type=str,
        default=None,
        help=f"Comma-list of source_jurisdiction codes for test split (default: "
        f"{','.join(DEFAULT_TEST_JURISDICTIONS)}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the manifest and print it; don't write files.",
    )
    return parser.parse_args(argv)


def _csv(arg: str | None) -> list[str]:
    if arg is None:
        return []
    return [s.strip() for s in arg.split(",") if s.strip()]


if __name__ == "__main__":
    sys.exit(main())
