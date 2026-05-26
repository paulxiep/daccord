"""Tier-5 CLI: extract per-framework citation registries from parsed markdown.

End-to-end flow:

1. Read `data/ingest/manifest.jsonl` (the tier-4 output manifest) and group
   rows by `framework`. PDPA-MY / PDPA-TH / DPA 2012 PH / BDSG each contribute
   two markdown files; others one.
2. For each framework (optionally allowlist-filtered with `--frameworks`):
   - Skip if `--skip-existing` (default) AND `data/registry/<framework>.json`
     exists AND all `source_sha256` match the current ingest manifest's
     `sha256_md`. Idempotency contract matches tier 4.
   - Read all input markdowns, dispatch to the per-framework extractor in
     `daccord.registry.patterns.FRAMEWORK_EXTRACTORS`.
   - Compute `cites_per_page` (citation_count / total page_count from ingest
     manifest) and `toy_gold_recall` against `data/gold/toy_v1.jsonl`.
   - Write `data/registry/<framework>.json` atomically.
   - Upsert into `data/registry/manifest.jsonl`.
3. Generate `data/registry/summary.md` (human-readable Markdown table).
4. Log one MLflow run under experiment `daccord-registry` with per-framework
   metrics (`citations_count__<fw>`, `cites_per_page__<fw>`,
   `toy_gold_recall__<fw>`).

M1 gate check: the script exits non-zero if any framework appearing in the
toy gold has `toy_gold_recall < 1.0`. CI / human reviewer can rely on the
exit code as the gate signal.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import mlflow

from daccord.ingest.manifest import IngestManifestEntry
from daccord.ingest.manifest import read_manifest as read_ingest_manifest
from daccord.registry.extract import compute_toy_gold_recall, extract_framework
from daccord.registry.schema import (
    RegistryManifestEntry,
    read_manifest,
    upsert,
    write_manifest,
    write_registry,
)
from daccord.tracking import compute_file_sha256, log_standard_params, setup_mlflow
from daccord.validation import validated

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INGEST_MANIFEST = REPO_ROOT / "data" / "ingest" / "manifest.jsonl"
DEFAULT_INGEST_ROOT = REPO_ROOT / "data" / "ingest"
DEFAULT_REGISTRY_DIR = REPO_ROOT / "data" / "registry"
DEFAULT_TOY_GOLD = REPO_ROOT / "data" / "gold" / "toy_v1.jsonl"
EXPERIMENT_NAME = "daccord-registry"
PROJECT_SEED = 42

log = logging.getLogger("extract_registry")


@validated
def group_by_framework(
    ingest_rows: list[IngestManifestEntry],
) -> dict[str, list[IngestManifestEntry]]:
    """Group successful ingest rows by `framework`, sorted by pdf_relpath.

    Failed rows are dropped — a framework that failed to parse can't
    contribute to its registry. The caller is responsible for reporting that
    drop separately.
    """
    grouped: dict[str, list[IngestManifestEntry]] = defaultdict(list)
    for row in ingest_rows:
        if row.failed or row.md_relpath is None:
            continue
        grouped[row.framework].append(row)
    return {fw: sorted(rows, key=lambda r: r.pdf_relpath) for fw, rows in grouped.items()}


@validated
def should_skip(
    framework: str,
    ingest_rows: list[IngestManifestEntry],
    registry_path: Path,
    prior_manifest: list[RegistryManifestEntry],
    skip_existing: bool,
) -> bool:
    """True iff `--skip-existing` AND a prior run still applies.

    A prior run applies when (a) the registry file exists, (b) the prior
    manifest row for this framework lists the same source_sha256 set as the
    current ingest manifest (in any order).
    """
    if not skip_existing or not registry_path.exists():
        return False
    matching = [e for e in prior_manifest if e.framework == framework]
    if not matching:
        return False
    last = matching[-1]
    prior_shas = set(last.source_sha256)
    current_shas = {row.sha256_md for row in ingest_rows if row.sha256_md is not None}
    return prior_shas == current_shas


@validated
def process_framework(
    framework: str,
    ingest_rows: list[IngestManifestEntry],
    registry_dir: Path,
    ingest_root: Path,
    toy_gold_path: Path,
) -> RegistryManifestEntry:
    """Read markdowns for `framework`, extract citations, write the registry JSON.

    Returns the manifest row for this framework — the caller upserts it into
    the registry manifest and may also feed its metrics to MLflow.
    """
    md_texts: list[str] = []
    source_documents: list[str] = []
    source_sha256: list[str] = []
    total_pages = 0
    jurisdiction = ingest_rows[0].jurisdiction
    for row in ingest_rows:
        assert row.md_relpath is not None  # guaranteed by group_by_framework
        assert row.sha256_md is not None
        md_path = REPO_ROOT / row.md_relpath
        if not md_path.exists():
            # Manifest references an md file that's gone — surface clearly.
            raise FileNotFoundError(
                f"manifest references missing markdown: {md_path} "
                f"(framework={framework}); re-run tier 4 ingest"
            )
        md_texts.append(md_path.read_text(encoding="utf-8"))
        source_documents.append(row.md_relpath)
        source_sha256.append(row.sha256_md)
        if row.page_count is not None:
            total_pages += row.page_count

    registry = extract_framework(
        framework_id=framework,
        jurisdiction=jurisdiction,
        md_texts=md_texts,
        source_documents=source_documents,
        source_sha256=source_sha256,
    )

    registry_path = registry_dir / f"{framework}.json"
    write_registry(registry_path, registry)
    sha256_registry = compute_file_sha256(registry_path)

    cites_per_page = registry.citation_count / total_pages if total_pages > 0 else None
    recall, missing = compute_toy_gold_recall(
        framework_id=framework,
        registry_ids=registry.citation_ids,
        toy_gold_path=toy_gold_path,
    )

    log.info(
        "[registry] %s count=%d cites/p=%s recall=%.2f missing=%s",
        framework,
        registry.citation_count,
        f"{cites_per_page:.2f}" if cites_per_page is not None else "n/a",
        recall,
        missing or "[]",
    )

    return RegistryManifestEntry(
        framework=framework,
        jurisdiction=jurisdiction,
        registry_relpath=registry_path.relative_to(REPO_ROOT).as_posix(),
        citation_count=registry.citation_count,
        cites_per_page=cites_per_page,
        toy_gold_recall=recall,
        toy_gold_missing=missing,
        sha256_registry=sha256_registry,
        source_documents=source_documents,
        source_sha256=source_sha256,
    )


@validated
def write_summary_md(path: Path, manifest_rows: list[RegistryManifestEntry]) -> None:
    """Write a human-readable Markdown table summarising every framework's run.

    No "generated at" timestamp — the file is committed and should be
    byte-identical across reruns. Git mtime carries the rest.
    """
    rows = sorted(manifest_rows, key=lambda r: r.framework)
    lines = [
        "# Tier-5 Citation Registries — Summary",
        "",
        "| Framework | Jurisdiction | Citations | Cites/Page | Toy-Gold Recall | Missing |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        cites_per_page = f"{r.cites_per_page:.2f}" if r.cites_per_page is not None else "n/a"
        recall = f"{r.toy_gold_recall:.2f}" if r.toy_gold_recall is not None else "n/a"
        missing = ", ".join(r.toy_gold_missing) if r.toy_gold_missing else "—"
        lines.append(
            f"| `{r.framework}` | `{r.jurisdiction}` | {r.citation_count} | "
            f"{cites_per_page} | {recall} | {missing} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


@validated
def run(
    ingest_manifest_path: Path,
    ingest_root: Path,
    registry_dir: Path,
    toy_gold_path: Path,
    frameworks: list[str] | None,
    skip_existing: bool,
    use_mlflow: bool,
) -> int:
    if not ingest_manifest_path.exists():
        log.error("ingest manifest not found: %s", ingest_manifest_path)
        return 2

    ingest_rows = read_ingest_manifest(ingest_manifest_path)
    grouped = group_by_framework(ingest_rows)
    if not grouped:
        log.error("no successful ingest rows in %s", ingest_manifest_path)
        return 2

    if frameworks:
        allowed = set(frameworks)
        grouped = {fw: rows for fw, rows in grouped.items() if fw in allowed}
        if not grouped:
            log.error("no frameworks match allowlist %s", frameworks)
            return 2

    registry_manifest_path = registry_dir / "manifest.jsonl"
    prior_manifest = read_manifest(registry_manifest_path)

    if use_mlflow:
        setup_mlflow(
            experiment_name=EXPERIMENT_NAME,
            tracking_uri=f"file:{(REPO_ROOT / 'mlruns').as_posix()}",
        )

    run_name = "registry-extract"
    manifest_state: list[RegistryManifestEntry] = list(prior_manifest)
    processed: list[RegistryManifestEntry] = []
    skipped: list[str] = []

    mlflow_ctx = mlflow.start_run(run_name=run_name) if use_mlflow else _Null()
    with mlflow_ctx:
        if use_mlflow:
            log_standard_params(
                run_name=run_name,
                seed=PROJECT_SEED,
                extra={
                    "frameworks_selected": ",".join(sorted(grouped.keys())),
                    "skip_existing": str(skip_existing),
                },
            )

        for framework in sorted(grouped.keys()):
            ingest_rows_for_fw = grouped[framework]
            registry_path = registry_dir / f"{framework}.json"
            if should_skip(
                framework,
                ingest_rows_for_fw,
                registry_path,
                prior_manifest,
                skip_existing,
            ):
                log.info("[registry] %s skip (cached)", framework)
                skipped.append(framework)
                continue

            row = process_framework(
                framework=framework,
                ingest_rows=ingest_rows_for_fw,
                registry_dir=registry_dir,
                ingest_root=ingest_root,
                toy_gold_path=toy_gold_path,
            )
            processed.append(row)
            manifest_state = upsert(manifest_state, row)
            write_manifest(registry_manifest_path, manifest_state)

            if use_mlflow:
                mlflow.log_metric(f"citations_count__{framework}", row.citation_count)
                if row.cites_per_page is not None:
                    mlflow.log_metric(f"cites_per_page__{framework}", row.cites_per_page)
                if row.toy_gold_recall is not None:
                    mlflow.log_metric(f"toy_gold_recall__{framework}", row.toy_gold_recall)

        write_summary_md(registry_dir / "summary.md", manifest_state)

        if use_mlflow:
            mlflow.log_metric("frameworks_processed", len(processed))
            mlflow.log_metric("frameworks_skipped", len(skipped))

    # M1 gate: any framework with toy-gold recall < 1.0 fails the gate.
    failures = [r for r in manifest_state if (r.toy_gold_recall or 0.0) < 1.0]
    if failures:
        log.error(
            "[done] M1 gate FAILED: %d framework(s) below 1.0 toy-gold recall:",
            len(failures),
        )
        for r in failures:
            log.error(
                "  %s: recall=%.2f missing=%s",
                r.framework,
                r.toy_gold_recall or 0.0,
                r.toy_gold_missing,
            )
        return 1

    log.info(
        "[done] processed=%d skipped=%d; manifest at %s",
        len(processed),
        len(skipped),
        registry_manifest_path,
    )
    return 0


class _Null:
    """No-op context manager for the `use_mlflow=False` path."""

    def __enter__(self) -> None: ...
    def __exit__(self, *_a: object) -> None: ...


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ingest-manifest", type=Path, default=DEFAULT_INGEST_MANIFEST)
    p.add_argument("--ingest-root", type=Path, default=DEFAULT_INGEST_ROOT)
    p.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    p.add_argument("--toy-gold", type=Path, default=DEFAULT_TOY_GOLD)
    p.add_argument(
        "--frameworks",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=None,
        help="comma-separated framework allowlist",
    )
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return run(
        ingest_manifest_path=args.ingest_manifest,
        ingest_root=args.ingest_root,
        registry_dir=args.registry_dir,
        toy_gold_path=args.toy_gold,
        frameworks=args.frameworks,
        skip_existing=not args.no_skip_existing,
        use_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    sys.exit(main())
