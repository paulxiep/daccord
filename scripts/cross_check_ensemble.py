"""Tier-6B+ CLI: bidirectional cross-check across all framework pairs.

For each forward pair (e.g. `gdpr__pdpa_sg`), compute whether each
tiered row's consensus is confirmed by the reverse direction
(`pdpa_sg__gdpr`). Writes
`data/ensemble/bidirectional/{forward_pair}.jsonl` for each pair and
prints per-pair status counts.

End-to-end flow:
  1. Discover `data/ensemble/tiered/*.jsonl` (tier-6B output).
  2. For each pair, call `compute_bidirectional_for_pair`:
     - Build the target framework's clause lookup from raw.
     - Resolve forward consensus → reverse source_id → reverse row.
     - Status = consistent / inconsistent / reverse_unknown /
       missing_in_registry / missing_reverse_row / missing_reverse_pair /
       no_forward_consensus.
  3. Write one JSONL per pair, sorted by source_id.

Output is idempotent — re-running on the same inputs produces
byte-identical files. The labeler (consumer/labeler/app.py) + splits
([scripts/build_splits.py](build_splits.py) with
`--promote-bidirectional-consistent`) consume this output.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

from daccord.ensemble.bidirectional import (
    BidirectionalResult,
    compute_bidirectional_for_pair,
)

log = logging.getLogger("cross_check_ensemble")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _discover_pairs(args.tiered_dir, args.framework_pair)
    if not pairs:
        log.error("No tiered files at %s/*.jsonl", args.tiered_dir)
        return 1

    log.info("Cross-checking %d pair(s)", len(pairs))
    overall: Counter[str] = Counter()
    promotable_by_tier: dict[str, Counter[str]] = {}

    for fp in pairs:
        results = compute_bidirectional_for_pair(
            forward_pair=fp,
            tiered_dir=args.tiered_dir,
            raw_dir=args.raw_dir,
        )
        _write_results(out_dir / f"{fp}.jsonl", results)
        counts = Counter(r.status for r in results)
        overall.update(counts)
        log.info(
            "  %-30s consistent=%-4d inconsistent=%-4d reverse_unknown=%-4d "
            "missing_in_registry=%-4d missing_reverse_row=%-4d "
            "missing_reverse_pair=%-4d no_forward_consensus=%-4d (total=%d)",
            fp,
            counts.get("consistent", 0),
            counts.get("inconsistent", 0),
            counts.get("reverse_unknown", 0),
            counts.get("missing_in_registry", 0),
            counts.get("missing_reverse_row", 0),
            counts.get("missing_reverse_pair", 0),
            counts.get("no_forward_consensus", 0),
            len(results),
        )

    log.info("")
    log.info("Overall:")
    for status in (
        "consistent",
        "inconsistent",
        "reverse_unknown",
        "missing_in_registry",
        "missing_reverse_row",
        "missing_reverse_pair",
        "no_forward_consensus",
    ):
        log.info("  %-22s = %d", status, overall.get(status, 0))
    log.info("")
    log.info(
        "Auto-promotable (status=consistent) — pass --promote-bidirectional-consistent "
        "to build_splits.py to include these MED rows in gold without hand-val: %d",
        overall.get("consistent", 0),
    )
    _ = promotable_by_tier
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
        "--raw-dir",
        type=Path,
        default=Path("data/ensemble/raw"),
        help="Directory of tier-7A raw candidates (used to build framework lookups).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/ensemble/bidirectional"),
        help="Output directory for {forward_pair}.jsonl files.",
    )
    parser.add_argument(
        "--framework-pair",
        type=str,
        default=None,
        help="Restrict to a single framework-pair (e.g. gdpr__pdpa_sg). "
        "If omitted, all pairs in --tiered-dir are processed.",
    )
    return parser.parse_args(argv)


def _discover_pairs(tiered_dir: Path, restrict_to: str | None) -> list[str]:
    if restrict_to is not None:
        path = tiered_dir / f"{restrict_to}.jsonl"
        return [restrict_to] if path.exists() else []
    return sorted(p.stem for p in tiered_dir.glob("*.jsonl"))


def _write_results(path: Path, rows: list[BidirectionalResult]) -> None:
    sorted_rows = sorted(rows, key=lambda r: r.source_id)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in sorted_rows:
            f.write(row.model_dump_json())
            f.write("\n")
    tmp.replace(path)


if __name__ == "__main__":
    sys.exit(main())
