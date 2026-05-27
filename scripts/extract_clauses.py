"""Tier-7A CLI: extract per-framework clause bodies from parsed markdown.

End-to-end flow:

1. Read `data/registry/manifest.jsonl` (tier-5 output) to learn each
   framework's source-document set + canonical citation_ids.
2. For each framework (optionally allowlist-filtered with `--frameworks`):
   - Read all source `.md` files referenced by the registry manifest.
   - Load the per-framework registry JSON for its canonical citation_ids.
   - Dispatch to `extract_framework_clauses(...)` to slice clause bodies
     between consecutive heading anchors.
   - Write `data/clauses/<framework>.json` atomically.
3. Log per-framework body_recall + missing IDs.

Idempotency: re-running on unchanged source markdown produces byte-identical
output. The script is fast (regex over ~1 MB of markdown per framework, well
under a second total).

Body recall < 1.0 is informational, NOT a gate failure. Tier 7A's smoke test
operates on toy_v1 gold (clause text already inline); the full-run path uses
`source_mechanism=clauses[citation_id]` when present and falls back to
`source_mechanism=citation_id` for missing entries.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from daccord.registry.clauses import extract_framework_clauses
from daccord.registry.schema import (
    read_manifest,
    read_registry,
    write_clauses,
)
from daccord.validation import validated

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_MANIFEST = REPO_ROOT / "data" / "registry" / "manifest.jsonl"
DEFAULT_REGISTRY_DIR = REPO_ROOT / "data" / "registry"
DEFAULT_CLAUSES_DIR = REPO_ROOT / "data" / "clauses"

log = logging.getLogger("extract_clauses")


@validated
def process_framework(
    framework: str,
    jurisdiction: str,
    source_documents: list[str],
    source_sha256: list[str],
    registry_citation_ids: list[str],
    clauses_dir: Path,
) -> tuple[int, float, int]:
    """Extract clauses for one framework and write `data/clauses/<fw>.json`.

    Returns `(extracted_count, body_recall, missing_count)` for logging.
    """
    md_texts: list[str] = []
    for relpath in source_documents:
        md_path = REPO_ROOT / relpath
        if not md_path.exists():
            raise FileNotFoundError(
                f"registry manifest references missing markdown: {md_path} "
                f"(framework={framework}); re-run tier 4 ingest"
            )
        md_texts.append(md_path.read_text(encoding="utf-8"))

    clauses = extract_framework_clauses(
        framework_id=framework,
        jurisdiction=jurisdiction,
        md_texts=md_texts,
        source_documents=source_documents,
        source_sha256=source_sha256,
        registry_citation_ids=registry_citation_ids,
    )

    clauses_path = clauses_dir / f"{framework}.json"
    write_clauses(clauses_path, clauses)
    return (len(clauses.clauses), clauses.body_recall, len(clauses.missing_citation_ids))


@validated
def run(
    registry_manifest_path: Path,
    registry_dir: Path,
    clauses_dir: Path,
    frameworks: list[str] | None,
) -> int:
    if not registry_manifest_path.exists():
        log.error("registry manifest not found: %s", registry_manifest_path)
        log.error("run `scripts/extract_registry.py` first to produce it.")
        return 2

    manifest_rows = read_manifest(registry_manifest_path)
    if not manifest_rows:
        log.error("registry manifest is empty: %s", registry_manifest_path)
        return 2

    if frameworks:
        allowed = set(frameworks)
        manifest_rows = [r for r in manifest_rows if r.framework in allowed]
        if not manifest_rows:
            log.error("no frameworks match allowlist %s", frameworks)
            return 2

    clauses_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    total_extracted = 0
    total_missing = 0
    low_recall: list[tuple[str, float]] = []

    for row in sorted(manifest_rows, key=lambda r: r.framework):
        registry_path = registry_dir / f"{row.framework}.json"
        if not registry_path.exists():
            log.warning("[clauses] %s skip (registry not on disk)", row.framework)
            continue
        registry = read_registry(registry_path)
        extracted, recall, missing = process_framework(
            framework=row.framework,
            jurisdiction=row.jurisdiction,
            source_documents=row.source_documents,
            source_sha256=row.source_sha256,
            registry_citation_ids=registry.citation_ids,
            clauses_dir=clauses_dir,
        )
        processed += 1
        total_extracted += extracted
        total_missing += missing
        if recall < 0.7:
            low_recall.append((row.framework, recall))
        log.info(
            "[clauses] %s extracted=%d recall=%.2f missing=%d",
            row.framework,
            extracted,
            recall,
            missing,
        )

    log.info(
        "[done] processed=%d frameworks; total_extracted=%d total_missing=%d",
        processed,
        total_extracted,
        total_missing,
    )
    if low_recall:
        log.warning(
            "[warn] %d framework(s) below 0.70 body recall — review manually:",
            len(low_recall),
        )
        for fw, rec in low_recall:
            log.warning("  %s: %.2f", fw, rec)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry-manifest", type=Path, default=DEFAULT_REGISTRY_MANIFEST)
    p.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    p.add_argument("--clauses-dir", type=Path, default=DEFAULT_CLAUSES_DIR)
    p.add_argument(
        "--frameworks",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=None,
        help="comma-separated framework allowlist (default: all)",
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return run(
        registry_manifest_path=args.registry_manifest,
        registry_dir=args.registry_dir,
        clauses_dir=args.clauses_dir,
        frameworks=args.frameworks,
    )


if __name__ == "__main__":
    sys.exit(main())
