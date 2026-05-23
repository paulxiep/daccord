from datetime import UTC, datetime
from pathlib import Path

from daccord.corpus.manifest import MANIFEST_SCHEMA_VERSION, Manifest, ManifestEntry


def _entry(framework: str, filename: str, sha: str = "a" * 64) -> ManifestEntry:
    return ManifestEntry(
        framework=framework,
        jurisdiction="eu",
        filename=filename,
        source_url="https://example.com/" + filename,
        local_path="data/raw/eu/" + framework + "/" + filename,
        sha256=sha,
        content_length=1234,
        retrieved_at=datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC),
        manual=False,
    )


def test_load_returns_empty_when_missing(tmp_path: Path) -> None:
    m = Manifest.load(tmp_path / "missing.json")
    assert m.schema_version == MANIFEST_SCHEMA_VERSION
    assert m.entries == []


def test_save_load_roundtrip_is_byte_stable(tmp_path: Path) -> None:
    fixed_ts = datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC)
    m = Manifest(
        generated_at=fixed_ts,
        entries=[_entry("gdpr", "z.pdf"), _entry("bdsg", "a.pdf")],
    )
    path = tmp_path / "manifest.json"
    m.save(path)
    text_first = path.read_text(encoding="utf-8")

    loaded = Manifest.load(path)
    loaded = loaded.model_copy(update={"generated_at": fixed_ts})
    loaded.save(path)
    text_second = path.read_text(encoding="utf-8")

    assert text_first == text_second


def test_save_sorts_entries_deterministically(tmp_path: Path) -> None:
    m = Manifest(
        generated_at=datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC),
        entries=[_entry("gdpr", "z.pdf"), _entry("bdsg", "a.pdf"), _entry("gdpr", "a.pdf")],
    )
    path = tmp_path / "manifest.json"
    m.save(path)
    reloaded = Manifest.load(path)
    keys = [e.key for e in reloaded.entries]
    assert keys == sorted(keys)


def test_upsert_replaces_existing() -> None:
    m = Manifest(
        generated_at=datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC),
        entries=[_entry("gdpr", "a.pdf", sha="b" * 64)],
    )
    m.upsert(_entry("gdpr", "a.pdf", sha="c" * 64))
    assert len(m.entries) == 1
    assert m.entries[0].sha256 == "c" * 64


def test_find_returns_none_when_absent() -> None:
    m = Manifest(generated_at=datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC), entries=[])
    assert m.find("gdpr", "missing.pdf") is None


def test_find_returns_matching_entry() -> None:
    entry = _entry("gdpr", "a.pdf")
    m = Manifest(generated_at=datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC), entries=[entry])
    found = m.find("gdpr", "a.pdf")
    assert found is not None
    assert found.sha256 == entry.sha256
