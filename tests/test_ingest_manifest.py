from datetime import UTC, datetime
from pathlib import Path

from daccord.ingest.manifest import (
    IngestManifestEntry,
    read_manifest,
    upsert,
    write_manifest,
)

_TS = datetime(2026, 5, 26, 12, 0, 0, tzinfo=UTC)


def _row(framework: str, pdf_relpath: str, *, failed: bool = False) -> IngestManifestEntry:
    return IngestManifestEntry(
        framework=framework,
        jurisdiction="eu",
        pdf_relpath=pdf_relpath,
        md_relpath=None if failed else pdf_relpath.replace(".pdf", ".md").replace("raw", "ingest"),
        page_count=None if failed else 10,
        char_count=None if failed else 12345,
        marker_version="1.10.2",
        parsed_at=_TS,
        seconds_elapsed=0.0 if failed else 12.3,
        sha256_pdf="a" * 64,
        sha256_md=None if failed else "b" * 64,
        failed=failed,
        error="boom" if failed else None,
    )


def test_read_missing_returns_empty(tmp_path: Path) -> None:
    assert read_manifest(tmp_path / "missing.jsonl") == []


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    entries = [_row("gdpr", "data/raw/eu/gdpr/x.pdf"), _row("bdsg", "data/raw/de/bdsg/y.pdf")]
    write_manifest(path, entries)
    reloaded = read_manifest(path)
    assert {(e.framework, e.pdf_relpath) for e in reloaded} == {
        (e.framework, e.pdf_relpath) for e in entries
    }


def test_write_sorts_by_key_deterministically(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest(
        path,
        [
            _row("gdpr", "data/raw/eu/gdpr/z.pdf"),
            _row("bdsg", "data/raw/de/bdsg/a.pdf"),
            _row("gdpr", "data/raw/eu/gdpr/a.pdf"),
        ],
    )
    keys = [(e.framework, e.pdf_relpath) for e in read_manifest(path)]
    assert keys == sorted(keys)


def test_upsert_replaces_by_key() -> None:
    a = _row("gdpr", "data/raw/eu/gdpr/x.pdf")
    b = _row("gdpr", "data/raw/eu/gdpr/x.pdf", failed=True)  # same key
    c = _row("bdsg", "data/raw/de/bdsg/y.pdf")
    out = upsert([a, c], b)
    assert len(out) == 2
    keys = {(e.framework, e.pdf_relpath, e.failed) for e in out}
    assert keys == {
        ("gdpr", "data/raw/eu/gdpr/x.pdf", True),
        ("bdsg", "data/raw/de/bdsg/y.pdf", False),
    }


def test_failed_row_serialises_with_nullable_fields(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [_row("uk_gdpr", "data/raw/uk/uk_gdpr/x.pdf", failed=True)])
    [reloaded] = read_manifest(path)
    assert reloaded.failed is True
    assert reloaded.md_relpath is None
    assert reloaded.page_count is None
    assert reloaded.sha256_md is None
    assert reloaded.error == "boom"
