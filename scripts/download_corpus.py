"""Download the regulatory PDF corpus declared in data/sources.yaml.

Idempotent: re-runs skip files whose sha256 already matches the manifest.
For sources marked manual, prints the expected drop-path and continues.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

from daccord.corpus.downloader import (
    DownloadResult,
    expected_path,
    fetch,
    hash_existing,
    make_client,
)
from daccord.corpus.manifest import Manifest, ManifestEntry
from daccord.corpus.sources import Source, SourcesSpec
from daccord.validation import validated

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCES = REPO_ROOT / "data" / "sources.yaml"
DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "raw"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "raw_manifest.json"

log = logging.getLogger("download_corpus")


class Outcome:
    DOWNLOADED = "downloaded"
    SKIPPED = "skipped_idempotent"
    MANUAL_PRESENT = "manual_present"
    MANUAL_MISSING = "manual_missing"
    ERROR = "error"


@validated
def _process_url_source(
    source: Source,
    raw_root: Path,
    manifest: Manifest,
    client: httpx.Client,
    dry_run: bool,
) -> tuple[str, ManifestEntry | None]:
    existing = manifest.find(source.framework, source.filename)
    on_disk = hash_existing(source, raw_root)
    fw, fn = source.framework, source.filename
    if existing and on_disk and existing.sha256 == on_disk.sha256:
        log.info("[skip ] %s/%s already present (sha matches manifest)", fw, fn)
        return Outcome.SKIPPED, existing
    if dry_run:
        log.info("[plan ] %s/%s -> %s", fw, fn, expected_path(source, raw_root))
        return Outcome.DOWNLOADED, None
    log.info("[fetch] %s/%s <- %s", fw, fn, source.url)
    result: DownloadResult = fetch(source, raw_root, client)
    return Outcome.DOWNLOADED, result.entry


@validated
def _process_manual_source(
    source: Source, raw_root: Path, manifest: Manifest, dry_run: bool
) -> tuple[str, ManifestEntry | None]:
    on_disk = hash_existing(source, raw_root)
    if on_disk is None:
        log.warning(
            "[MANUAL] %s/%s missing — place file at %s",
            source.framework,
            source.filename,
            expected_path(source, raw_root),
        )
        return Outcome.MANUAL_MISSING, None
    existing = manifest.find(source.framework, source.filename)
    if existing and existing.sha256 == on_disk.sha256:
        log.info(
            "[skip ] %s/%s manual file present (sha matches manifest)",
            source.framework,
            source.filename,
        )
        return Outcome.MANUAL_PRESENT, existing
    if dry_run:
        log.info("[plan ] %s/%s manual hash refresh", source.framework, source.filename)
        return Outcome.MANUAL_PRESENT, None
    log.info("[hash ] %s/%s manual file new/changed — recording", source.framework, source.filename)
    return Outcome.MANUAL_PRESENT, on_disk


@validated
def run(
    sources_path: Path,
    raw_root: Path,
    manifest_path: Path,
    frameworks: list[str] | None,
    dry_run: bool,
) -> int:
    spec = SourcesSpec.from_yaml(sources_path).filter_frameworks(frameworks)
    manifest = Manifest.load(manifest_path)
    raw_root.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    missing_manual: list[Path] = []
    errors: list[tuple[Source, Exception]] = []

    with make_client() as client:
        for source in spec.sources:
            try:
                if source.manual:
                    outcome, entry = _process_manual_source(source, raw_root, manifest, dry_run)
                    if outcome == Outcome.MANUAL_MISSING:
                        missing_manual.append(expected_path(source, raw_root))
                else:
                    outcome, entry = _process_url_source(
                        source, raw_root, manifest, client, dry_run
                    )
                counts[outcome] = counts.get(outcome, 0) + 1
                if entry is not None and not dry_run:
                    manifest.upsert(entry)
            except Exception as exc:
                log.exception("[ERROR] %s/%s: %s", source.framework, source.filename, exc)
                errors.append((source, exc))
                counts[Outcome.ERROR] = counts.get(Outcome.ERROR, 0) + 1

    if not dry_run:
        manifest = manifest.model_copy(update={"generated_at": datetime.now(UTC)})
        manifest.save(manifest_path)

    log.info("---- summary ----")
    for k in sorted(counts):
        log.info("%s: %d", k, counts[k])
    if missing_manual:
        log.info("manual files awaiting placement:")
        for p in missing_manual:
            log.info("  %s", p)
    if errors:
        log.error("encountered %d download errors", len(errors))
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--frameworks",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=None,
        help="comma-separated framework filter (e.g. gdpr,bdsg)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return run(
        sources_path=args.sources,
        raw_root=args.raw_root,
        manifest_path=args.manifest,
        frameworks=args.frameworks,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
