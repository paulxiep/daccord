"""Tier-6B CLI: tier ensemble candidates into HIGH/MED/LOW/SALVAGE buckets.

End-to-end flow:

1. Discover `data/ensemble/raw/{framework_pair}__{model}.jsonl` files in
   `--raw-dir` (default `data/ensemble/raw/`). If `--framework-pair` is
   given, restrict to that pair; otherwise process every pair found.
2. For each framework pair:
   - Optionally verify all `--expected-models` produced output for every
     source_id (catches partial runs).
   - Group candidates by `source_id`, run `classify_tier` per group.
   - Write `data/ensemble/tiered/{framework_pair}.jsonl` atomically.
3. Print per-pair counts: HIGH / MED / LOW / SALVAGE.

Reproducibility: rerunning on the same inputs produces byte-identical
output (sorted by source_id; deterministic tier function).

Tier-8 input: this script's output is what the hand-validation reviewer
opens. HIGH-tier rows are promoted (with 100% spot-check on a stratified
sample); MED/LOW/SALVAGE are 100%-validated.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

from daccord.ensemble import TieredPair, tier_framework_pair

log = logging.getLogger("tier_ensemble")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    raw_dir = args.raw_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _discover_framework_pairs(raw_dir, args.framework_pair)
    if not pairs:
        log.error("No framework pairs found at %s", raw_dir)
        return 1

    log.info("Tiering %d framework-pair(s): %s", len(pairs), ", ".join(pairs))

    overall: Counter[str] = Counter()
    for fp in pairs:
        tiered = tier_framework_pair(
            framework_pair=fp,
            raw_dir=raw_dir,
            expected_models=args.expected_models or None,
        )
        _write_tiered(out_dir / f"{fp}.jsonl", tiered)
        counts = Counter(t.tier for t in tiered)
        overall.update(counts)
        log.info(
            "  %s → HIGH=%d MED=%d LOW=%d SALVAGE=%d (total=%d)",
            fp,
            counts.get("HIGH", 0),
            counts.get("MED", 0),
            counts.get("LOW", 0),
            counts.get("SALVAGE", 0),
            len(tiered),
        )

    log.info(
        "Overall: HIGH=%d MED=%d LOW=%d SALVAGE=%d",
        overall.get("HIGH", 0),
        overall.get("MED", 0),
        overall.get("LOW", 0),
        overall.get("SALVAGE", 0),
    )
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/ensemble/raw"),
        help="Directory containing {framework_pair}__{model}.jsonl files (tier 7A output).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/ensemble/tiered"),
        help="Output directory for {framework_pair}.jsonl tiered files.",
    )
    parser.add_argument(
        "--framework-pair",
        type=str,
        default=None,
        help="Restrict to a single framework-pair (e.g. gdpr__pdpa_sg). "
        "If omitted, all pairs in --raw-dir are processed.",
    )
    parser.add_argument(
        "--expected-models",
        type=str,
        nargs="*",
        default=None,
        help="Fail if any source_id is missing a vote from any of these models. "
        "Pass once per model, e.g. --expected-models llama-4-scout llama-4-maverick "
        "claude-haiku-4-5 gemini-3-1-flash.",
    )
    return parser.parse_args(argv)


def _discover_framework_pairs(raw_dir: Path, restrict_to: str | None) -> list[str]:
    if restrict_to is not None:
        if list(raw_dir.glob(f"{restrict_to}__*.jsonl")):
            return [restrict_to]
        return []
    # Strip the "__{model}.jsonl" suffix to recover the framework_pair.
    pairs: set[str] = set()
    for path in raw_dir.glob("*__*.jsonl"):
        stem = path.stem  # e.g. gdpr__pdpa_sg__llama-4-scout
        # Take everything before the LAST "__"
        idx = stem.rfind("__")
        if idx <= 0:
            continue
        pairs.add(stem[:idx])
    return sorted(pairs)


def _write_tiered(path: Path, rows: list[TieredPair]) -> None:
    # Sort by source_id for deterministic output.
    sorted_rows = sorted(rows, key=lambda r: r.source_id)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in sorted_rows:
            f.write(row.model_dump_json())
            f.write("\n")
    tmp.replace(path)


if __name__ == "__main__":
    sys.exit(main())
