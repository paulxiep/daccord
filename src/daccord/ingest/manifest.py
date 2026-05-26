"""On-disk schema for `data/ingest/manifest.jsonl` — tier-4's per-document record.

JSONL (not JSON) because the manifest is append-friendly during long parse
runs: a per-row write keeps progress visible to `docker compose logs -f`
without re-serialising the whole list. The file is atomic-replaced at end of
run, but partial-run inspection is the more common debug path.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

from daccord.validation import ValidatedModel, validated


class IngestManifestEntry(ValidatedModel):
    """One row of `data/ingest/manifest.jsonl` — one PDF→markdown conversion."""

    framework: str
    jurisdiction: str
    pdf_relpath: str
    md_relpath: str | None
    page_count: int | None
    char_count: int | None
    marker_version: str
    parsed_at: datetime
    seconds_elapsed: float
    sha256_pdf: str
    sha256_md: str | None
    failed: bool = False
    error: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        """Stable identity for upsert / dedupe: (framework, pdf_relpath)."""
        return (self.framework, self.pdf_relpath)


@validated
def read_manifest(path: Path) -> list[IngestManifestEntry]:
    """Return all entries from `path` (JSONL). Missing file → empty list."""
    if not path.exists():
        return []
    out: list[IngestManifestEntry] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            out.append(IngestManifestEntry.model_validate_json(stripped))
    return out


@validated
def write_manifest(path: Path, entries: list[IngestManifestEntry]) -> None:
    """Atomic-write `entries` to `path` as JSONL.

    Entries are sorted by (framework, pdf_relpath) for deterministic diffs.
    Uses a temp file + `replace` so concurrent readers never see a half-written
    file. UTF-8, `ensure_ascii=False` so Thai/French characters in error
    messages round-trip readably.
    """
    sorted_entries = sorted(entries, key=lambda e: e.key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
        suffix=".jsonl.tmp",
    ) as tmp:
        for entry in sorted_entries:
            tmp.write(entry.model_dump_json() + "\n")
        tmp_name = tmp.name
    Path(tmp_name).replace(path)


@validated
def upsert(
    entries: list[IngestManifestEntry], new: IngestManifestEntry
) -> list[IngestManifestEntry]:
    """Return a new list with any entry sharing `new.key` replaced by `new`."""
    return [e for e in entries if e.key != new.key] + [new]


@validated
def now_utc() -> datetime:
    """Return the current UTC time. Wrapped so tests can monkeypatch it."""
    return datetime.now(UTC)
