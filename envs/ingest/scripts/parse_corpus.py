"""Tier-4 CLI: parse all corpus PDFs to markdown via Marker.

End-to-end flow:

1. Read `data/raw_manifest.json` (the corpus manifest from 1D).
2. Filter to `--subset {toy,full}` plus optional `--frameworks <ids>` allowlist.
3. Build the Marker `PdfConverter` once (heavy — pays the model-load cost
   exactly once per process).
4. For each entry: skip if the output markdown already exists and the input
   sha256 matches the prior manifest row (idempotent re-runs are no-ops).
5. Parse, write `data/ingest/<jur>/<framework>/<file>.md`, append a row to
   `data/ingest/manifest.jsonl` (rewritten atomically after each doc so a
   crash mid-sweep loses at most one doc's progress).
6. Per-doc failure (Marker exception) is recorded in the manifest as
   `failed=true` + error string; the sweep continues — one bad
   browser-print PDF (R8) must not kill the run.
7. Log one MLflow run under experiment `daccord-ingest` with per-doc
   metrics (`ingest_seconds__<framework>`, `ingest_chars__<framework>`).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import mlflow

from daccord.corpus.manifest import Manifest, ManifestEntry
from daccord.ingest.manifest import (
    IngestManifestEntry,
    now_utc,
    read_manifest,
    upsert,
    write_manifest,
)
from daccord.ingest.marker_runner import (
    DocumentOutput,
    make_converter,
    parse_document,
    parser_version,
)
from daccord.tracking import compute_file_sha256, log_standard_params, setup_mlflow
from daccord.validation import validated

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_MANIFEST = REPO_ROOT / "data" / "raw_manifest.json"
DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "raw"
DEFAULT_INGEST_DIR = REPO_ROOT / "data" / "ingest"
DEFAULT_INGEST_MANIFEST = DEFAULT_INGEST_DIR / "manifest.jsonl"
EXPERIMENT_NAME = "daccord-ingest"
PROJECT_SEED = 42

# Toy subset: three small native-text English regulator PDFs. Total ~1.4 MB,
# expected wall-time <3 min. Picked to surface schema/wrapper/Marker-version
# errors cheap before kicking off the 13-doc sweep.
TOY_FRAMEWORKS: frozenset[str] = frozenset({"gdpr", "bdsg", "pdpa_sg"})
# bdsg has two PDFs (DE + EN); the toy run pins the EN one for fast iteration.
TOY_FILENAMES: frozenset[str] = frozenset(
    {"reg_2016_679_consolidated.pdf", "bdsg_en_current.pdf", "pdpa_sg_current.pdf"}
)

log = logging.getLogger("parse_corpus")


@validated
def select_entries(
    raw_manifest: Manifest,
    subset: str,
    frameworks: list[str] | None,
) -> list[ManifestEntry]:
    """Filter `raw_manifest.entries` by `--subset` and optional framework allowlist.

    `subset='toy'` picks `TOY_FILENAMES` (3 small EN PDFs). `subset='full'`
    returns every entry. The optional `--frameworks` allowlist is an
    additional AND-filter on top, useful for targeted re-runs.
    """
    if subset == "toy":
        picked = [e for e in raw_manifest.entries if e.filename in TOY_FILENAMES]
    elif subset == "full":
        picked = list(raw_manifest.entries)
    else:
        raise ValueError(f"unknown subset {subset!r} (expected 'toy' or 'full')")
    if frameworks:
        allowed = set(frameworks)
        picked = [e for e in picked if e.framework in allowed]
    return picked


@validated
def output_paths(entry: ManifestEntry, raw_root: Path, ingest_root: Path) -> tuple[Path, Path]:
    """Return `(pdf_abs_path, md_abs_path)` for `entry`.

    The output md mirrors the raw layout: `data/raw/<jur>/<framework>/<file>.pdf`
    → `data/ingest/<jur>/<framework>/<file>.md`. Tier 5's registry extractor
    iterates the manifest, not the directory, so layout drift here would be
    caught at registry-build time.
    """
    pdf_abs = raw_root / entry.jurisdiction / entry.framework / entry.filename
    md_filename = Path(entry.filename).with_suffix(".md").name
    md_abs = ingest_root / entry.jurisdiction / entry.framework / md_filename
    return pdf_abs, md_abs


@validated
def should_skip(
    entry: ManifestEntry,
    pdf_abs: Path,
    md_abs: Path,
    prior: list[IngestManifestEntry],
    skip_existing: bool,
) -> bool:
    """True iff `--skip-existing` is on AND a prior successful run still applies.

    A prior row applies when (a) the markdown file still exists, (b) the input
    PDF's sha256 hasn't changed since the prior parse. Failed prior runs are
    NOT skipped — we want to retry them.
    """
    if not skip_existing:
        return False
    if not md_abs.exists():
        return False
    pdf_rel = pdf_abs.relative_to(REPO_ROOT).as_posix()
    matching = [e for e in prior if e.framework == entry.framework and e.pdf_relpath == pdf_rel]
    if not matching:
        return False
    last = matching[-1]
    if last.failed:
        return False
    return last.sha256_pdf == entry.sha256


@validated
def parse_one(
    entry: ManifestEntry,
    converter: object,
    raw_root: Path,
    ingest_root: Path,
    marker_v: str,
) -> IngestManifestEntry:
    """Parse one PDF; return a manifest row (success or failure).

    Marker exceptions are caught and recorded as `failed=true` so the caller's
    sweep continues — one bad UK/FR browser-print PDF (R8) must not kill the
    whole 13-doc run. `converter` is typed `object` here because it's an
    opaque Marker handle; `parse_document` narrows it back to `Any`.
    """
    pdf_abs, md_abs = output_paths(entry, raw_root, ingest_root)
    pdf_rel = pdf_abs.relative_to(REPO_ROOT).as_posix()
    md_rel = md_abs.relative_to(REPO_ROOT).as_posix()

    try:
        output: DocumentOutput = parse_document(pdf_abs, md_abs, converter)
    except Exception as exc:  # noqa: BLE001 — intentional: isolate per-doc failures
        log.exception("[ingest] %s FAILED: %s", entry.framework, exc)
        return IngestManifestEntry(
            framework=entry.framework,
            jurisdiction=entry.jurisdiction,
            pdf_relpath=pdf_rel,
            md_relpath=None,
            page_count=None,
            char_count=None,
            marker_version=marker_v,
            parsed_at=now_utc(),
            seconds_elapsed=0.0,
            sha256_pdf=entry.sha256,
            sha256_md=None,
            failed=True,
            error=f"{type(exc).__name__}: {exc}",
        )

    log.info(
        "[ingest] %s %s → %s (%dp, %dc, %.1fs)",
        entry.framework,
        pdf_rel,
        md_rel,
        output.page_count,
        output.char_count,
        output.seconds_elapsed,
    )
    return IngestManifestEntry(
        framework=entry.framework,
        jurisdiction=entry.jurisdiction,
        pdf_relpath=pdf_rel,
        md_relpath=md_rel,
        page_count=output.page_count,
        char_count=output.char_count,
        marker_version=marker_v,
        parsed_at=now_utc(),
        seconds_elapsed=output.seconds_elapsed,
        sha256_pdf=entry.sha256,
        sha256_md=compute_file_sha256(md_abs),
        failed=False,
        error=None,
    )


@validated
def run(
    raw_manifest_path: Path,
    raw_root: Path,
    ingest_root: Path,
    ingest_manifest_path: Path,
    subset: str,
    frameworks: list[str] | None,
    skip_existing: bool,
    use_mlflow: bool,
) -> int:
    if not raw_manifest_path.exists():
        log.error("raw manifest not found: %s", raw_manifest_path)
        return 2

    raw_manifest = Manifest.load(raw_manifest_path)
    selected = select_entries(raw_manifest, subset, frameworks)
    if not selected:
        log.error("no entries match subset=%s frameworks=%s", subset, frameworks)
        return 2

    log.info(
        "[ingest] subset=%s selected %d/%d entries",
        subset,
        len(selected),
        len(raw_manifest.entries),
    )

    prior = read_manifest(ingest_manifest_path)

    if use_mlflow:
        # Pin tracking-URI to repo-root mlruns/ — script runs from envs/ingest/
        # so the default file:./mlruns would land in envs/ingest/mlruns/.
        setup_mlflow(
            experiment_name=EXPERIMENT_NAME,
            tracking_uri=f"file:{(REPO_ROOT / 'mlruns').as_posix()}",
        )

    run_name = f"ingest-{subset}"
    marker_v = parser_version()

    converter: object | None = None
    manifest_state: list[IngestManifestEntry] = list(prior)
    parsed_this_run: list[IngestManifestEntry] = []

    mlflow_ctx = mlflow.start_run(run_name=run_name) if use_mlflow else _Null()
    with mlflow_ctx:
        if use_mlflow:
            log_standard_params(
                run_name=run_name,
                seed=PROJECT_SEED,
                extra={
                    "subset": subset,
                    "marker_version": marker_v,
                    "selected_count": str(len(selected)),
                },
            )

        for entry in selected:
            pdf_abs, md_abs = output_paths(entry, raw_root, ingest_root)
            if should_skip(entry, pdf_abs, md_abs, prior, skip_existing):
                log.info(
                    "[ingest] %s skip (cached: %s)",
                    entry.framework,
                    md_abs.relative_to(REPO_ROOT).as_posix(),
                )
                continue
            if not pdf_abs.exists():
                log.error("[ingest] %s missing PDF: %s", entry.framework, pdf_abs)
                continue

            if converter is None:
                log.info("[ingest] loading Marker model dict (one-time)…")
                converter = make_converter()
                log.info("[ingest] Marker ready (version=%s)", marker_v)

            row = parse_one(entry, converter, raw_root, ingest_root, marker_v)
            parsed_this_run.append(row)
            manifest_state = upsert(manifest_state, row)
            # Persist after every doc so a crash loses at most one doc's progress.
            write_manifest(ingest_manifest_path, manifest_state)

            if use_mlflow and not row.failed:
                metric_suffix = f"{entry.framework}__{entry.filename}"
                mlflow.log_metric(f"ingest_seconds__{metric_suffix}", row.seconds_elapsed)
                if row.char_count is not None:
                    mlflow.log_metric(f"ingest_chars__{metric_suffix}", row.char_count)

        succeeded = [r for r in parsed_this_run if not r.failed]
        failed = [r for r in parsed_this_run if r.failed]
        log.info(
            "[done] %d parsed (%d ok, %d failed); manifest at %s",
            len(parsed_this_run),
            len(succeeded),
            len(failed),
            ingest_manifest_path,
        )
        if use_mlflow:
            mlflow.log_metric("docs_succeeded", len(succeeded))
            mlflow.log_metric("docs_failed", len(failed))
            mlflow.log_metric("docs_parsed", len(parsed_this_run))

    return 0 if not failed else 1


class _Null:
    """No-op context manager for the `use_mlflow=False` path."""

    def __enter__(self) -> None: ...
    def __exit__(self, *_a: object) -> None: ...


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset", choices=["toy", "full"], default="toy")
    p.add_argument(
        "--frameworks",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=None,
        help="comma-separated framework allowlist (AND-applied on top of --subset)",
    )
    p.add_argument("--raw-manifest", type=Path, default=DEFAULT_RAW_MANIFEST)
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    p.add_argument("--ingest-dir", type=Path, default=DEFAULT_INGEST_DIR)
    p.add_argument("--ingest-manifest", type=Path, default=DEFAULT_INGEST_MANIFEST)
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="force re-parse even if md exists and input sha256 matches prior run",
    )
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return run(
        raw_manifest_path=args.raw_manifest,
        raw_root=args.raw_root,
        ingest_root=args.ingest_dir,
        ingest_manifest_path=args.ingest_manifest,
        subset=args.subset,
        frameworks=args.frameworks,
        skip_existing=not args.no_skip_existing,
        use_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    sys.exit(main())
