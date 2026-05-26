"""Tests for the tier-4 corpus-parse CLI logic.

Heavy Marker call (`parse_one`) is exercised via a monkeypatched
`parse_document` that either returns a stub `DocumentOutput` or raises — both
paths must produce a manifest row so the sweep continues past per-doc failures.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from daccord.corpus.manifest import Manifest, ManifestEntry
from daccord.ingest.manifest import IngestManifestEntry


def _entry(framework: str, jurisdiction: str, filename: str, sha: str = "a" * 64) -> ManifestEntry:
    return ManifestEntry(
        framework=framework,
        jurisdiction=jurisdiction,
        filename=filename,
        source_url=None,
        local_path=f"data/raw/{jurisdiction}/{framework}/{filename}",
        sha256=sha,
        content_length=1000,
        retrieved_at=datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC),
        manual=False,
    )


def test_select_entries_toy_picks_three_known_files() -> None:
    import parse_corpus

    raw = Manifest(
        generated_at=datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC),
        entries=[
            _entry("gdpr", "eu", "reg_2016_679_consolidated.pdf"),
            _entry("bdsg", "de", "bdsg_en_current.pdf"),
            _entry("bdsg", "de", "bdsg_de_current.pdf"),
            _entry("pdpa_sg", "sg", "pdpa_sg_current.pdf"),
            _entry("uk_gdpr", "uk", "uk_gdpr_current.pdf"),
        ],
    )
    picked = parse_corpus.select_entries(raw, subset="toy", frameworks=None)
    assert {(e.framework, e.filename) for e in picked} == {
        ("gdpr", "reg_2016_679_consolidated.pdf"),
        ("bdsg", "bdsg_en_current.pdf"),
        ("pdpa_sg", "pdpa_sg_current.pdf"),
    }


def test_select_entries_full_returns_all() -> None:
    import parse_corpus

    raw = Manifest(
        generated_at=datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC),
        entries=[_entry("gdpr", "eu", "x.pdf"), _entry("bdsg", "de", "y.pdf")],
    )
    picked = parse_corpus.select_entries(raw, subset="full", frameworks=None)
    assert len(picked) == 2


def test_select_entries_framework_allowlist_intersects_with_subset() -> None:
    import parse_corpus

    raw = Manifest(
        generated_at=datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC),
        entries=[
            _entry("gdpr", "eu", "reg_2016_679_consolidated.pdf"),
            _entry("bdsg", "de", "bdsg_en_current.pdf"),
            _entry("pdpa_sg", "sg", "pdpa_sg_current.pdf"),
        ],
    )
    picked = parse_corpus.select_entries(raw, subset="toy", frameworks=["gdpr"])
    assert [e.framework for e in picked] == ["gdpr"]


def test_output_paths_mirror_raw_layout(tmp_path: Path) -> None:
    import parse_corpus

    e = _entry("gdpr", "eu", "x.pdf")
    pdf, md = parse_corpus.output_paths(e, tmp_path / "raw", tmp_path / "ingest")
    assert pdf == tmp_path / "raw" / "eu" / "gdpr" / "x.pdf"
    assert md == tmp_path / "ingest" / "eu" / "gdpr" / "x.md"


def test_should_skip_true_when_md_exists_and_sha_matches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import parse_corpus

    # Redirect REPO_ROOT so the relative-path comparison inside should_skip
    # operates against tmp_path, not the real repo.
    monkeypatch.setattr(parse_corpus, "REPO_ROOT", tmp_path)

    e = _entry("gdpr", "eu", "x.pdf", sha="d" * 64)
    pdf, md = parse_corpus.output_paths(e, tmp_path / "raw", tmp_path / "ingest")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("dummy", encoding="utf-8")
    prior = [
        IngestManifestEntry(
            framework="gdpr",
            jurisdiction="eu",
            pdf_relpath="raw/eu/gdpr/x.pdf",
            md_relpath="ingest/eu/gdpr/x.md",
            page_count=1,
            char_count=10,
            marker_version="1.10.2",
            parsed_at=datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC),
            seconds_elapsed=1.0,
            sha256_pdf="d" * 64,
            sha256_md="b" * 64,
        )
    ]
    assert parse_corpus.should_skip(e, pdf, md, prior, skip_existing=True) is True


def test_should_skip_false_when_sha_changed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import parse_corpus

    monkeypatch.setattr(parse_corpus, "REPO_ROOT", tmp_path)
    e = _entry("gdpr", "eu", "x.pdf", sha="e" * 64)  # new sha
    pdf, md = parse_corpus.output_paths(e, tmp_path / "raw", tmp_path / "ingest")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("dummy", encoding="utf-8")
    prior = [
        IngestManifestEntry(
            framework="gdpr",
            jurisdiction="eu",
            pdf_relpath="raw/eu/gdpr/x.pdf",
            md_relpath="ingest/eu/gdpr/x.md",
            page_count=1,
            char_count=10,
            marker_version="1.10.2",
            parsed_at=datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC),
            seconds_elapsed=1.0,
            sha256_pdf="d" * 64,  # old sha
            sha256_md="b" * 64,
        )
    ]
    assert parse_corpus.should_skip(e, pdf, md, prior, skip_existing=True) is False


def test_should_skip_false_when_prior_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import parse_corpus

    monkeypatch.setattr(parse_corpus, "REPO_ROOT", tmp_path)
    e = _entry("uk_gdpr", "uk", "uk_gdpr_current.pdf", sha="d" * 64)
    pdf, md = parse_corpus.output_paths(e, tmp_path / "raw", tmp_path / "ingest")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("dummy", encoding="utf-8")
    prior = [
        IngestManifestEntry(
            framework="uk_gdpr",
            jurisdiction="uk",
            pdf_relpath="raw/uk/uk_gdpr/uk_gdpr_current.pdf",
            md_relpath=None,
            page_count=None,
            char_count=None,
            marker_version="1.10.2",
            parsed_at=datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC),
            seconds_elapsed=0.0,
            sha256_pdf="d" * 64,
            sha256_md=None,
            failed=True,
            error="boom",
        )
    ]
    assert parse_corpus.should_skip(e, pdf, md, prior, skip_existing=True) is False


def test_should_skip_false_when_skip_existing_off(tmp_path: Path) -> None:
    import parse_corpus

    e = _entry("gdpr", "eu", "x.pdf")
    pdf, md = parse_corpus.output_paths(e, tmp_path / "raw", tmp_path / "ingest")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("dummy", encoding="utf-8")
    assert parse_corpus.should_skip(e, pdf, md, prior=[], skip_existing=False) is False


def test_should_skip_false_when_md_missing(tmp_path: Path) -> None:
    import parse_corpus

    e = _entry("gdpr", "eu", "x.pdf")
    pdf, md = parse_corpus.output_paths(e, tmp_path / "raw", tmp_path / "ingest")
    # md does NOT exist
    assert parse_corpus.should_skip(e, pdf, md, prior=[], skip_existing=True) is False


def test_parse_one_records_failure_when_marker_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import parse_corpus

    monkeypatch.setattr(parse_corpus, "REPO_ROOT", tmp_path)

    def boom(*_a: object, **_kw: object) -> object:
        raise RuntimeError("marker exploded")

    monkeypatch.setattr(parse_corpus, "parse_document", boom)

    e = _entry("uk_gdpr", "uk", "uk_gdpr_current.pdf")
    row = parse_corpus.parse_one(
        entry=e,
        converter=object(),
        raw_root=tmp_path / "raw",
        ingest_root=tmp_path / "ingest",
        marker_v="1.10.2-fake",
    )
    assert row.failed is True
    assert row.error is not None
    assert "marker exploded" in row.error
    assert row.md_relpath is None
    assert row.sha256_md is None


def test_parse_one_records_success_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import parse_corpus

    from daccord.ingest.marker_runner import DocumentOutput

    pdf_dir = tmp_path / "raw" / "eu" / "gdpr"
    pdf_dir.mkdir(parents=True)
    pdf = pdf_dir / "reg_2016_679_consolidated.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    def fake_parse(pdf_path: Path, out_md_path: Path, _converter: object) -> DocumentOutput:
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        out_md_path.write_text("# stub markdown\n", encoding="utf-8")
        return DocumentOutput(
            pdf_path=pdf_path,
            md_path=out_md_path,
            markdown="# stub markdown\n",
            char_count=17,
            page_count=99,
            seconds_elapsed=0.25,
        )

    monkeypatch.setattr(parse_corpus, "parse_document", fake_parse)
    # The function uses REPO_ROOT to compute relpaths; redirect it to tmp_path
    # so the test stays self-contained.
    monkeypatch.setattr(parse_corpus, "REPO_ROOT", tmp_path)

    e = _entry("gdpr", "eu", "reg_2016_679_consolidated.pdf")
    row = parse_corpus.parse_one(
        entry=e,
        converter=object(),
        raw_root=tmp_path / "raw",
        ingest_root=tmp_path / "ingest",
        marker_v="1.10.2-fake",
    )
    assert row.failed is False
    assert row.page_count == 99
    assert row.char_count == 17
    assert row.md_relpath == "ingest/eu/gdpr/reg_2016_679_consolidated.md"
    assert row.sha256_md is not None
    assert len(row.sha256_md) == 64
