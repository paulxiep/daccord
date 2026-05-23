from pathlib import Path

import pytest
from pydantic import ValidationError

from daccord.corpus.sources import Source, SourcesSpec


def test_source_accepts_url_only() -> None:
    s = Source(
        framework="gdpr",
        jurisdiction="eu",
        filename="reg_2016_679.pdf",
        description="GDPR consolidated",
        url="https://eur-lex.europa.eu/example.pdf",  # type: ignore[arg-type]
    )
    assert s.manual is False
    assert s.url is not None


def test_source_accepts_manual_only() -> None:
    s = Source(
        framework="pdpa_th",
        jurisdiction="th",
        filename="royal_gazette_2022_01.pdf",
        description="amendment",
        manual=True,
    )
    assert s.manual is True
    assert s.url is None


def test_source_rejects_both_url_and_manual() -> None:
    with pytest.raises(ValidationError):
        Source(
            framework="gdpr",
            jurisdiction="eu",
            filename="x.pdf",
            description="bad",
            url="https://example.com/x.pdf",  # type: ignore[arg-type]
            manual=True,
        )


def test_source_rejects_neither_url_nor_manual() -> None:
    with pytest.raises(ValidationError):
        Source(framework="gdpr", jurisdiction="eu", filename="x.pdf", description="bad")


def test_sources_spec_from_yaml_roundtrip(tmp_path: Path) -> None:
    yaml_text = """
sources:
  - framework: gdpr
    jurisdiction: eu
    filename: reg_2016_679.pdf
    description: GDPR consolidated
    url: https://eur-lex.europa.eu/example.pdf
  - framework: pdpa_th
    jurisdiction: th
    filename: royal_gazette_2022_01.pdf
    description: amendment placeholder
    manual: true
"""
    p = tmp_path / "sources.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    spec = SourcesSpec.from_yaml(p)
    assert len(spec.sources) == 2
    assert spec.sources[0].framework == "gdpr"
    assert spec.sources[1].manual is True


def test_filter_frameworks_subset() -> None:
    spec = SourcesSpec(
        sources=[
            Source(
                framework="gdpr",
                jurisdiction="eu",
                filename="a.pdf",
                description="x",
                url="https://example.com/a.pdf",  # type: ignore[arg-type]
            ),
            Source(
                framework="bdsg",
                jurisdiction="de",
                filename="b.pdf",
                description="y",
                url="https://example.com/b.pdf",  # type: ignore[arg-type]
            ),
        ]
    )
    filtered = spec.filter_frameworks(["bdsg"])
    assert [s.framework for s in filtered.sources] == ["bdsg"]


def test_filter_frameworks_none_returns_all() -> None:
    spec = SourcesSpec(
        sources=[
            Source(
                framework="gdpr",
                jurisdiction="eu",
                filename="a.pdf",
                description="x",
                url="https://example.com/a.pdf",  # type: ignore[arg-type]
            ),
        ]
    )
    assert spec.filter_frameworks(None).sources == spec.sources
