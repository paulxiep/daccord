"""Tier-7A clause-body extractor tests.

Each framework gets a small fixture mimicking the real parsed Marker output
shape; the assertion is that bodies slice correctly between heading anchors
and that canonical IDs match the registry's key space.

Edge cases: heading at end of doc, multiple paragraphs between headings,
cross-references that should NOT split a body, letter suffixes (26D, 38a),
PDPA-TH preamble cross-refs (must be skipped).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from daccord.registry.clauses import (
    EXTRACTOR_VERSION,
    FRAMEWORK_CLAUSE_EXTRACTORS,
    extract_clauses_bdsg,
    extract_clauses_dpa_2012_ph,
    extract_clauses_dpa_2018,
    extract_clauses_gdpr_like,
    extract_clauses_pdpa_my,
    extract_clauses_pdpa_sg,
    extract_clauses_pdpa_th,
    extract_framework_clauses,
)
from daccord.registry.schema import (
    FrameworkClauses,
    read_clauses,
    write_clauses,
)


class TestExtractGdprLike:
    def test_basic_three_articles(self) -> None:
        md = (
            "#### *Article 1*\n#### **Subject-matter**\n\n"
            "Body of article one. Two sentences.\n\n"
            "# *Article 2*\n#### **Material scope**\n\n"
            "Body of article two.\n\n"
            "### *Article 3*\n\n"
            "Body of article three. Final EOF body slice.\n"
        )
        out = extract_clauses_gdpr_like(md)
        assert set(out.keys()) == {"1", "2", "3"}
        assert "Body of article one" in out["1"]
        assert "Body of article two" in out["2"]
        assert "Body of article three" in out["3"]
        assert "Final EOF body slice" in out["3"]

    def test_cross_ref_in_body_does_not_split(self) -> None:
        """Inline `Article N` in body text must NOT count as a section heading."""
        md = (
            "#### *Article 1*\n\n"
            "Body that mentions Article 2 and Article 99 inline.\n\n"
            "# *Article 2*\n\n"
            "Real article two body.\n"
        )
        out = extract_clauses_gdpr_like(md)
        assert set(out.keys()) == {"1", "2"}
        assert "mentions Article 2" in out["1"]
        assert "Real article two" in out["2"]

    def test_heading_at_eof_no_body(self) -> None:
        """Trailing heading with no body must be skipped (empty body)."""
        md = "#### *Article 1*\n\nBody one.\n\n# *Article 99*\n"
        out = extract_clauses_gdpr_like(md)
        assert "1" in out and "Body one" in out["1"]
        assert "99" not in out  # heading present but no body — dropped


class TestExtractDpa2018:
    def test_bare_number_bold_headings(self) -> None:
        md = (
            "### **1 Overview**\n\nOverview body.\n\n"
            "### **2 Protection of personal data**\n\nProtection body.\n\n"
            "### **9A Processing in reliance on relevant international law**\n\n"
            "9A body with letter suffix.\n"
        )
        out = extract_clauses_dpa_2018(md)
        assert set(out.keys()) == {"1", "2", "9A"}
        assert "Overview body" in out["1"]
        assert "9A body" in out["9A"]


class TestExtractBdsg:
    def test_de_paragraph_sign(self) -> None:
        md = (
            "#### **§ 1 Anwendungsbereich**\n\nGerman body for section 1.\n\n"
            "### **§ 3 Verarbeitung**\n\nBody for section 3.\n\n"
            "#### **§ 38a**\n\nLetter-suffix body.\n"
        )
        out = extract_clauses_bdsg(md)
        assert "1" in out and "German body for section 1" in out["1"]
        assert "3" in out and "Body for section 3" in out["3"]
        assert "38A" in out and "Letter-suffix body" in out["38A"]

    def test_en_bold_section(self) -> None:
        md = (
            "**Section 15 Activity reports** Body of section 15. Continued.\n\n"
            "**Section 16 Compensation** Body of section 16.\n"
        )
        out = extract_clauses_bdsg(md)
        assert "15" in out and "Body of section 15" in out["15"]
        assert "16" in out and "Body of section 16" in out["16"]


class TestExtractPdpaMy:
    def test_bilingual_seksyen_then_section(self) -> None:
        md = (
            "## **Seksyen 1. Tajuk ringkas dan permulaan kuat kuasa.**\n\n"
            "BM body for section 1.\n\n"
            "## **Section 1. Short title.**\n\n"
            "EN body for section 1.\n\n"
            "## **Seksyen 2. Pemakaian**\n\n"
            "BM body for section 2.\n"
        )
        out = extract_clauses_pdpa_my(md)
        assert set(out.keys()) == {"1", "2"}
        # First-occurrence-wins so the BM body wins on section 1.
        assert "BM body for section 1" in out["1"]
        assert "BM body for section 2" in out["2"]


class TestExtractPdpaSg:
    def test_bold_numbered_sections(self) -> None:
        md = (
            "- 1. Short title\n- 2. Interpretation\n"
            "Body content from TOC.\n\n"
            "**1.** This Act is the Personal Data Protection Act 2012.\n\n"
            "**3.** The purpose of this Act is to govern.\n\n"
            "**26D.**—(1) Where an organisation assesses that a breach is notifiable.\n"
        )
        out = extract_clauses_pdpa_sg(md)
        assert set(out.keys()) == {"1", "3", "26D"}
        assert "This Act is the Personal Data Protection Act" in out["1"]
        assert "Where an organisation assesses" in out["26D"]


class TestExtractDpa2012Ph:
    def test_section_and_sec_dot_forms(self) -> None:
        md = (
            "SECTION 1*. Short Title. –* This Act shall be known as the Data Privacy Act 2012.\n\n"
            "SEC. 2. *Declaration of Policy. –* It is the policy of the State.\n\n"
            "- SEC. 5. *Protection Afforded to Journalists. –* Nothing in this Act.\n"
        )
        out = extract_clauses_dpa_2012_ph(md)
        assert set(out.keys()) == {"1", "2", "5"}
        assert "Short Title" in out["1"]
        assert "Declaration of Policy" in out["2"]
        assert "Journalists" in out["5"]


class TestExtractPdpaTh:
    def test_thai_matra_with_preamble_skipped(self) -> None:
        md = (
            "พระราชบัญญัตินี้มีบทบัญญัติบางประการเกี่ยวกับการจำกัดสิทธิ "
            "ซึ่งมาตรา ๒๖ ประกอบกับมาตรา ๓๒ ของรัฐธรรมนูญ\n\n"
            'มาตรา ๑ พระราชบัญญัตินี้เรียกว่า "พระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล"\n\n'
            "มาตรา ๒ พระราชบัญญัตินี้ให้ใช้บังคับ\n\n"
            "มาตรา ๓ ในกรณีที่มีกฎหมายว่าด้วยการใด\n"
        )
        out = extract_clauses_pdpa_th(md)
        # The preamble's มาตรา ๒๖ / ๓๒ refs must NOT appear — those came
        # before the first มาตรา ๑ and were trimmed.
        assert "26" not in out
        assert "32" not in out
        assert set(out.keys()) == {"1", "2", "3"}
        assert "พระราชบัญญัตินี้เรียกว่า" in out["1"]

    def test_english_fallback_when_no_thai(self) -> None:
        md = (
            "**Section 1** This Act is called the Personal Data Protection Act.\n\n"
            "### **Section 4** This Act shall not apply.\n\n"
            "**Section 5** This Act applies to processing.\n"
        )
        out = extract_clauses_pdpa_th(md)
        assert set(out.keys()) == {"1", "4", "5"}
        assert "Personal Data Protection Act" in out["1"]


class TestExtractFrameworkClauses:
    def test_registry_filter_drops_extra_matches(self) -> None:
        """Anchor matches not in the registry's citation_ids get dropped."""
        md = (
            "#### *Article 1*\n\nBody one.\n\n"
            "# *Article 2*\n\nBody two.\n\n"
            "### *Article 999*\n\nBody nine-nine-nine (not in registry).\n"
        )
        out = extract_framework_clauses(
            framework_id="gdpr",
            jurisdiction="eu",
            md_texts=[md],
            source_documents=["fixture.md"],
            source_sha256=["deadbeef"],
            registry_citation_ids=["1", "2"],
        )
        assert isinstance(out, FrameworkClauses)
        assert set(out.clauses.keys()) == {"1", "2"}
        assert out.body_recall == 1.0
        assert out.missing_citation_ids == []

    def test_body_recall_reports_missing_headings(self) -> None:
        """Registry has IDs whose headings Marker dropped → body_recall < 1."""
        md = "#### *Article 1*\n\nBody one.\n\n# *Article 3*\n\nBody three.\n"
        out = extract_framework_clauses(
            framework_id="gdpr",
            jurisdiction="eu",
            md_texts=[md],
            source_documents=["fixture.md"],
            source_sha256=["deadbeef"],
            registry_citation_ids=["1", "2", "3"],
        )
        assert set(out.clauses.keys()) == {"1", "3"}
        assert out.missing_citation_ids == ["2"]
        assert out.body_recall == pytest.approx(2 / 3)

    def test_pdpa_th_prefers_english_over_thai(self) -> None:
        """When both languages are present, EN body overrides Thai for same ID."""
        thai_md = "มาตรา ๑ พระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล\n"
        en_md = "**Section 1** This Act is called the Personal Data Protection Act.\n"
        out = extract_framework_clauses(
            framework_id="pdpa_th",
            jurisdiction="th",
            md_texts=[thai_md, en_md],
            source_documents=["pdpa_th_thai.md", "pdpa_th_english.md"],
            source_sha256=["aaa", "bbb"],
            registry_citation_ids=["1"],
        )
        # EN body wins per the language-preference ordering inside the aggregator.
        assert "Personal Data Protection Act" in out.clauses["1"]

    def test_unknown_framework_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown_framework"):
            extract_framework_clauses(
                framework_id="unknown_framework",
                jurisdiction="??",
                md_texts=[""],
                source_documents=["x.md"],
                source_sha256=["x"],
                registry_citation_ids=[],
            )

    def test_mismatched_input_lists_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            extract_framework_clauses(
                framework_id="gdpr",
                jurisdiction="eu",
                md_texts=["a", "b"],
                source_documents=["x.md"],
                source_sha256=["x", "y"],
                registry_citation_ids=[],
            )

    def test_dispatch_table_covers_nine_frameworks(self) -> None:
        assert set(FRAMEWORK_CLAUSE_EXTRACTORS.keys()) == {
            "gdpr",
            "uk_gdpr",
            "loi_il",
            "dpa_2018",
            "bdsg",
            "pdpa_my",
            "pdpa_sg",
            "dpa_2012_ph",
            "pdpa_th",
        }


class TestSchemaRoundTrip:
    def test_write_and_read_clauses_round_trip(self, tmp_path: Path) -> None:
        clauses = FrameworkClauses(
            framework="gdpr",
            jurisdiction="eu",
            clauses={"1": "Body of article one.", "2": "Body of article two."},
            body_recall=1.0,
            missing_citation_ids=[],
            source_documents=["data/ingest/eu/gdpr/reg_2016_679_consolidated.md"],
            source_sha256=["deadbeef"],
            extractor_version=EXTRACTOR_VERSION,
        )
        path = tmp_path / "gdpr.json"
        write_clauses(path, clauses)
        loaded = read_clauses(path)
        assert loaded == clauses

    def test_write_clauses_pretty_printed(self, tmp_path: Path) -> None:
        clauses = FrameworkClauses(
            framework="gdpr",
            jurisdiction="eu",
            clauses={"1": "Body."},
            body_recall=1.0,
            missing_citation_ids=[],
            source_documents=["a.md"],
            source_sha256=["x"],
            extractor_version=EXTRACTOR_VERSION,
        )
        path = tmp_path / "gdpr.json"
        write_clauses(path, clauses)
        raw = path.read_text(encoding="utf-8")
        # 2-space indent + trailing newline matches `write_registry`'s contract.
        assert raw.startswith("{\n  ")
        assert raw.endswith("\n")
        # Round-trip via json.loads ensures valid JSON.
        parsed = json.loads(raw)
        assert parsed["framework"] == "gdpr"
