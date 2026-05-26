"""On-disk schema for tier-5 citation registries.

Two artifacts per run:

  - `data/registry/<framework_id>.json` — one `FrameworkRegistry` per framework
    family (9 files for the MVP scope). Contains the canonical citation IDs
    (e.g. "6", "26d", "38a") + parallel display IDs (e.g. "Article 6",
    "Section 26D", "§ 38a"), plus source-document SHA256s for cache
    invalidation.
  - `data/registry/manifest.jsonl` — one `RegistryManifestEntry` per framework.
    Append-friendly JSONL (same idiom as `data/ingest/manifest.jsonl`) so a
    partial-run inspection is straightforward. The manifest carries the
    per-framework summary metrics (count, density, toy-gold recall) the M1
    gate is checked against.

Canonical-ID contract: every `citation_id` is the output of
`daccord.eval.scoring.normalize_citation_id` applied to the raw heading text.
That function is M0-locked — registry hits and eval Tier-1 citation_match
hits MUST share key space, otherwise the registry can't gate ensemble output
or tag provenance at serve time.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from daccord.validation import ValidatedModel, validated


class FrameworkRegistry(ValidatedModel):
    """One framework's valid citation IDs — the file at data/registry/<id>.json.

    `citation_ids` and `display_ids` are parallel lists: same length, same
    order. `citation_ids[i]` is the canonical form (post-normalize_citation_id),
    `display_ids[i]` is the original-language form for human inspection /
    summary reports.

    Intentionally has NO timestamp field: the file is committed to git and
    reruns must produce byte-identical output (`extractor_version` +
    `source_sha256` carry all audit info; git mtime + log give the rest).
    """

    framework: str
    jurisdiction: str
    citation_ids: list[str]
    display_ids: list[str]
    source_documents: list[str]
    source_sha256: list[str]
    extractor_version: str
    citation_count: int


class RegistryManifestEntry(ValidatedModel):
    """One row of data/registry/manifest.jsonl — per-framework run summary.

    Like the registry payload, no timestamp field — reruns must be
    byte-identical so committed manifests don't churn on every CI build.
    """

    framework: str
    jurisdiction: str
    registry_relpath: str
    citation_count: int
    cites_per_page: float | None
    toy_gold_recall: float | None
    toy_gold_missing: list[str]
    sha256_registry: str
    source_documents: list[str]
    source_sha256: list[str]

    @property
    def key(self) -> tuple[str]:
        """Stable identity for upsert / dedupe: (framework,)."""
        return (self.framework,)


@validated
def read_manifest(path: Path) -> list[RegistryManifestEntry]:
    """Return all entries from `path` (JSONL). Missing file → empty list."""
    if not path.exists():
        return []
    out: list[RegistryManifestEntry] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            out.append(RegistryManifestEntry.model_validate_json(stripped))
    return out


@validated
def write_manifest(path: Path, entries: list[RegistryManifestEntry]) -> None:
    """Atomic-write `entries` to `path` as JSONL, sorted by framework.

    Same temp-file + `replace` pattern as `daccord.ingest.manifest.write_manifest`.
    UTF-8, `ensure_ascii=False` so non-ASCII (e.g. Thai display IDs) round-trip.
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
    entries: list[RegistryManifestEntry], new: RegistryManifestEntry
) -> list[RegistryManifestEntry]:
    """Return a new list with any entry sharing `new.key` replaced by `new`."""
    return [e for e in entries if e.key != new.key] + [new]


@validated
def write_registry(path: Path, registry: FrameworkRegistry) -> None:
    """Atomic-write `registry` to `path` as pretty-printed JSON.

    Pretty-printed (2-space indent) because the file is human-inspected during
    M1 spot-checking — `data/registry/gdpr.json` should be diffable on GitHub
    without horizontal scrolling. Atomicity via temp file + `replace`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = registry.model_dump(mode="json")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
        suffix=".json.tmp",
    ) as tmp:
        json.dump(payload, tmp, indent=2, ensure_ascii=False, sort_keys=False)
        tmp.write("\n")
        tmp_name = tmp.name
    Path(tmp_name).replace(path)


@validated
def read_registry(path: Path) -> FrameworkRegistry:
    """Load a framework registry JSON; raise FileNotFoundError if missing."""
    return FrameworkRegistry.model_validate_json(path.read_text(encoding="utf-8"))
