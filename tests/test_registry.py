"""Tier-5 citation-registry tests.

Covers:
  - patterns.py: per-framework extractors against representative fixtures
    drawn from the real parsed markdown shape (Marker output for each law).
  - extract.py: dedupe across multiple input documents, toy-gold recall.
  - schema.py: atomic JSON / JSONL round-trips + upsert semantics.

Fixture markdown is intentionally small + verbatim-style — the regex must
work on what Marker actually emits, not on synthetic perfect text. If
Marker version bumps change emission style, these fixtures should change too.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from daccord.registry.extract import compute_toy_gold_recall, extract_framework
from daccord.registry.patterns import (
    FRAMEWORK_EXTRACTORS,
    base_section,
    extract_article_en,
    extract_bdsg,
    extract_dpa_2012_ph,
    extract_dpa_2018,
    extract_pdpa_my,
    extract_pdpa_th,
    extract_section_en,
)
from daccord.registry.schema import (
    FrameworkRegistry,
    RegistryManifestEntry,
    read_manifest,
    read_registry,
    upsert,
    write_manifest,
    write_registry,
)

# ─────────────────────────────────────────────────────────────────────────────
# Per-framework extractor tests (one per language style).
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractArticleEn:
    def test_gdpr_style_headings(self) -> None:
        md = (
            "#### *Article 1*\n#### **Subject-matter**\n\n"
            "# *Article 2*\n#### **Material scope**\n\n"
            "### *Article 3*\n\n"
            "# *CHAPTER II*\n\n# *Article 5*\n"
        )
        pairs = extract_article_en(md)
        canon = [p[0] for p in pairs]
        assert canon == ["1", "2", "3", "5"]
        display = [p[1] for p in pairs]
        assert display == ["Article 1", "Article 2", "Article 3", "Article 5"]

    def test_handles_high_numbers_without_overlap(self) -> None:
        md = "Article 9. Article 99. Article 199."
        pairs = extract_article_en(md)
        # "Article 9" + "Article 99" + "Article 199" — three distinct IDs,
        # no overlap from the greedy \d+ + \b combo.
        assert [p[0] for p in pairs] == ["9", "99", "199"]

    def test_ignores_lowercase_cross_refs(self) -> None:
        # FR body text uses lowercase "à l'article 3 de la présente loi";
        # we only want capital-A headings to avoid double-counting.
        md = "**Article 1**\nbody mentioning à l'article 3 here.\n**Article 2**\n"
        pairs = extract_article_en(md)
        assert [p[0] for p in pairs] == ["1", "2"]


class TestExtractSectionEn:
    def test_pdpa_sg_letter_suffixes(self) -> None:
        md = "- 13. Consent required\n- 15A. Deemed consent by notification\n- 26D. Notification."
        # The bullet form alone doesn't carry "Section" — but the PDPA-SG
        # markdown also has a `#### Section` heading before the list and
        # `Section 26D` body refs. Use a more representative fixture:
        md = (
            "#### Section\n- 13. Consent required\n- 15A. Deemed consent\n"
            "Section 26A defines a notifiable breach. Section 26D requires notification."
        )
        pairs = extract_section_en(md)
        canon = [p[0] for p in pairs]
        # Letter-suffix uppercase normalized, sorted by numeric prefix.
        assert "26A" in canon
        assert "26D" in canon

    def test_no_section_marker_yields_empty(self) -> None:
        md = "No section markers here, just prose."
        assert extract_section_en(md) == []


class TestExtractDpa2012Ph:
    def test_mixed_section_forms(self) -> None:
        md = (
            "SECTION 1. Short Title.\n"
            "SEC. 2. Declaration of Policy.\n"
            "SEC. 12. Criteria for Lawful Processing.\n"
            "Section 20 covers security.\n"
        )
        pairs = extract_dpa_2012_ph(md)
        canon = [p[0] for p in pairs]
        assert canon == ["1", "2", "12", "20"]


class TestExtractDpa2018:
    def test_bare_heading_and_body_crossref(self) -> None:
        md = (
            "### **1 Overview**\n"
            "- (1) This Act makes provision\n"
            "### **2 Protection of personal data**\n"
            "S. 1 in force at Royal Assent\n"
            "s. 45(2) confers...\n"
            "### **45 Right of access**\n"
            "### **69 Designation of DPO**\n"
        )
        pairs = extract_dpa_2018(md)
        canon = [p[0] for p in pairs]
        # 1, 2, 45, 69 — both heading and body-cross-ref forms collapse to
        # the same canonical ID.
        assert set(canon) >= {"1", "2", "45", "69"}


class TestExtractBdsg:
    def test_merges_paragraph_sign_and_section(self) -> None:
        # The DE+EN union: DE PDF uses § N, EN translation uses Section N.
        md_de = "§ 1 Anwendungsbereich\n§ 38 Datenschutzbeauftragte\n§ 38a Sonderfälle"
        md_en = "Section 1 Scope\nSection 38 Data protection officers"
        # Single string with both encodings — mimics merging across two
        # ingest entries.
        pairs = extract_bdsg(md_de + "\n" + md_en)
        canon = [p[0] for p in pairs]
        # "38a" stays as "38A" after _canonicalize (letter suffix uppercased).
        # Lowercase "a" preserved here because _canonicalize re-uppercases.
        assert "1" in canon
        assert "38" in canon
        assert "38A" in canon


class TestExtractPdpaMy:
    def test_merges_seksyen_and_section(self) -> None:
        md = (
            "BAHAGIAN I PERMULAAN\n"
            "Seksyen 1 Tajuk ringkas\n"
            "Seksyen 6 Prinsip Am\n"
            "Section 30 Right to access personal data\n"
        )
        pairs = extract_pdpa_my(md)
        canon = [p[0] for p in pairs]
        # Malay + English both contribute, deduped: {1, 6, 30}.
        assert canon == ["1", "6", "30"]


class TestExtractPdpaTh:
    def test_thai_numerals_normalized(self) -> None:
        md = "มาตรา ๑ พระราชบัญญัตินี้\nมาตรา ๙๓ ห้าม\nSection 37 covers security\n"
        pairs = extract_pdpa_th(md)
        canon = [p[0] for p in pairs]
        # Thai ๑/๙๓ + Arabic 37, merged + sorted numerically.
        assert canon == ["1", "37", "93"]

    def test_thai_arabic_mixed_dedupes(self) -> None:
        # Both encodings of section 93 should collapse to one entry.
        md = "มาตรา ๙๓ first form\nมาตรา 93 second form\nSection 93 english\n"
        pairs = extract_pdpa_th(md)
        assert [p[0] for p in pairs] == ["93"]


class TestBaseSection:
    def test_strips_subsection_parens(self) -> None:
        assert base_section("6(1)(a)") == "6"
        assert base_section("37(4)") == "37"
        assert base_section("12(a)") == "12"

    def test_preserves_letter_suffix(self) -> None:
        assert base_section("26D") == "26D"
        assert base_section("15A") == "15A"

    def test_uppercases_lowercase_suffix(self) -> None:
        assert base_section("38a") == "38A"

    def test_empty_input_returns_empty(self) -> None:
        assert base_section("") == ""


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch table integrity.
# ─────────────────────────────────────────────────────────────────────────────


def test_all_9_mvp_frameworks_have_extractors() -> None:
    """Catches a typo'd or missing entry in FRAMEWORK_EXTRACTORS."""
    expected = {
        "gdpr",
        "uk_gdpr",
        "dpa_2018",
        "loi_il",
        "pdpa_sg",
        "dpa_2012_ph",
        "bdsg",
        "pdpa_my",
        "pdpa_th",
    }
    assert set(FRAMEWORK_EXTRACTORS.keys()) == expected


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration tests — extract_framework + compute_toy_gold_recall.
# ─────────────────────────────────────────────────────────────────────────────


def test_extract_framework_dedupes_across_documents() -> None:
    md_a = "Article 1\nArticle 5\nArticle 10"
    md_b = "Article 5\nArticle 99"
    reg = extract_framework(
        framework_id="gdpr",
        jurisdiction="eu",
        md_texts=[md_a, md_b],
        source_documents=["a.md", "b.md"],
        source_sha256=["a" * 64, "b" * 64],
    )
    assert reg.citation_ids == ["1", "5", "10", "99"]
    assert reg.citation_count == 4
    assert reg.source_documents == ["a.md", "b.md"]
    assert reg.extractor_version == "tier-5/v1"


def test_extract_framework_unknown_id_raises() -> None:
    with pytest.raises(KeyError, match="no extractor registered"):
        extract_framework(
            framework_id="nope",
            jurisdiction="??",
            md_texts=["..."],
            source_documents=["x.md"],
            source_sha256=["x" * 64],
        )


def test_extract_framework_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="same length"):
        extract_framework(
            framework_id="gdpr",
            jurisdiction="eu",
            md_texts=["a", "b"],
            source_documents=["only-one.md"],
            source_sha256=["x" * 64],
        )


def test_compute_toy_gold_recall_strips_subsections(tmp_path: Path) -> None:
    # Gold cites "Article 6(1)(a)" → base "6"; registry has "6" → recall=1.0.
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(
        json.dumps(
            {
                "id": "t1",
                "source_jurisdiction": "eu",
                "source_framework": "gdpr",
                "source_citation_id": "Article 6(1)(a)",
                "source_mechanism": "Lawful processing on consent.",
                "source_language": "en",
                "target_jurisdiction": "uk",
                "target_framework": "uk_gdpr",
                "target_citation_id": "Article 6(1)(a)",
                "target_mechanism": "UK retained.",
                "target_language": "en",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    recall, missing = compute_toy_gold_recall(
        framework_id="gdpr",
        registry_ids=["5", "6", "7"],
        toy_gold_path=gold_path,
    )
    assert recall == 1.0
    assert missing == []


def test_compute_toy_gold_recall_flags_missing(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(
        json.dumps(
            {
                "id": "t2",
                "source_jurisdiction": "sg",
                "source_framework": "pdpa_sg",
                "source_citation_id": "Section 26D",
                "source_mechanism": "Breach notification.",
                "source_language": "en",
                "target_jurisdiction": "eu",
                "target_framework": "gdpr",
                "target_citation_id": "Article 33",
                "target_mechanism": "Breach.",
                "target_language": "en",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    recall, missing = compute_toy_gold_recall(
        framework_id="pdpa_sg",
        registry_ids=["1", "13", "21"],  # 26D missing
        toy_gold_path=gold_path,
    )
    assert recall == 0.0
    assert missing == ["26D"]


def test_compute_toy_gold_recall_missing_file_returns_one(tmp_path: Path) -> None:
    # Defensive: if the toy gold file isn't present, treat as no-data → pass.
    recall, missing = compute_toy_gold_recall(
        framework_id="gdpr",
        registry_ids=["1"],
        toy_gold_path=tmp_path / "missing.jsonl",
    )
    assert recall == 1.0
    assert missing == []


def test_compute_toy_gold_recall_framework_not_in_gold_returns_one(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(
        json.dumps(
            {
                "id": "t3",
                "source_jurisdiction": "eu",
                "source_framework": "gdpr",
                "source_citation_id": "Article 5",
                "source_mechanism": "Principles.",
                "source_language": "en",
                "target_jurisdiction": "sg",
                "target_framework": "pdpa_sg",
                "target_citation_id": "Section 24",
                "target_mechanism": "Protection.",
                "target_language": "en",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    recall, missing = compute_toy_gold_recall(
        framework_id="loi_il",  # not in this gold file
        registry_ids=["1"],
        toy_gold_path=gold_path,
    )
    assert recall == 1.0
    assert missing == []


# ─────────────────────────────────────────────────────────────────────────────
# Schema I/O tests — atomic JSON / JSONL round-trips.
# ─────────────────────────────────────────────────────────────────────────────


def _make_registry(framework: str = "gdpr") -> FrameworkRegistry:
    return FrameworkRegistry(
        framework=framework,
        jurisdiction="eu",
        citation_ids=["1", "2", "5", "99"],
        display_ids=["Article 1", "Article 2", "Article 5", "Article 99"],
        source_documents=["data/ingest/eu/gdpr/x.md"],
        source_sha256=["a" * 64],
        extractor_version="tier-5/v1",
        citation_count=4,
    )


def _make_manifest_row(framework: str = "gdpr") -> RegistryManifestEntry:
    return RegistryManifestEntry(
        framework=framework,
        jurisdiction="eu",
        registry_relpath=f"data/registry/{framework}.json",
        citation_count=4,
        cites_per_page=1.5,
        toy_gold_recall=1.0,
        toy_gold_missing=[],
        sha256_registry="r" * 64,
        source_documents=["data/ingest/eu/gdpr/x.md"],
        source_sha256=["a" * 64],
    )


def test_write_and_read_registry_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "gdpr.json"
    reg = _make_registry()
    write_registry(path, reg)
    loaded = read_registry(path)
    assert loaded.citation_ids == reg.citation_ids
    assert loaded.display_ids == reg.display_ids
    assert loaded.citation_count == 4


def test_write_registry_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "gdpr.json"
    reg = _make_registry()
    write_registry(path, reg)
    first_bytes = path.read_bytes()
    write_registry(path, reg)
    second_bytes = path.read_bytes()
    assert first_bytes == second_bytes


def test_manifest_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    rows = [_make_manifest_row("gdpr"), _make_manifest_row("bdsg")]
    write_manifest(path, rows)
    loaded = read_manifest(path)
    assert {r.framework for r in loaded} == {"gdpr", "bdsg"}


def test_manifest_sorts_by_framework(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest(
        path,
        [
            _make_manifest_row("pdpa_sg"),
            _make_manifest_row("bdsg"),
            _make_manifest_row("gdpr"),
        ],
    )
    loaded = read_manifest(path)
    assert [r.framework for r in loaded] == ["bdsg", "gdpr", "pdpa_sg"]


def test_upsert_replaces_by_framework() -> None:
    a = _make_manifest_row("gdpr")
    b = _make_manifest_row("gdpr")  # same key
    b_modified = b.model_copy(update={"citation_count": 999})
    c = _make_manifest_row("bdsg")
    out = upsert([a, c], b_modified)
    assert len(out) == 2
    by_fw = {r.framework: r for r in out}
    assert by_fw["gdpr"].citation_count == 999
    assert by_fw["bdsg"].citation_count == 4


def test_read_missing_manifest_returns_empty(tmp_path: Path) -> None:
    assert read_manifest(tmp_path / "missing.jsonl") == []
