"""Unit tests for `daccord.bakeoff.scoring`.

Covers the citation regex on Thai-script + Arabic-numeral inputs, the
precision/recall arithmetic for the four meaningful boundary cases (perfect
match, partial overlap, empty expected, empty extracted), and the per-parser
aggregator's `None` handling.
"""

from daccord.bakeoff.scoring import (
    aggregate,
    extract_citations,
    normalize_thai_numerals,
    score_page,
)


def test_normalize_thai_numerals_maps_digits() -> None:
    assert normalize_thai_numerals("๙๓") == "93"
    assert normalize_thai_numerals("มาตรา ๑๒๓") == "มาตรา 123"
    # Latin chars pass through untouched
    assert normalize_thai_numerals("Article 32") == "Article 32"


def test_extract_citations_finds_thai_numeral_sections() -> None:
    md = "ตามมาตรา ๒๔ และมาตรา ๒๖ ของพระราชบัญญัตินี้"
    assert extract_citations(md) == ["มาตรา 24", "มาตรา 26"]


def test_extract_citations_finds_arabic_numeral_sections() -> None:
    md = "Section reference: มาตรา 30 and มาตรา 32."
    assert extract_citations(md) == ["มาตรา 30", "มาตรา 32"]


def test_extract_citations_dedups_and_sorts() -> None:
    md = "มาตรา ๓ ... มาตรา ๓ again ... มาตรา ๑."
    assert extract_citations(md) == ["มาตรา 1", "มาตรา 3"]


def test_extract_citations_ignores_subsections() -> None:
    # มาตรา ๒๔ วรรค ๑ — only the section number itself is the citation.
    md = "มาตรา ๒๔ วรรคหนึ่ง"
    assert extract_citations(md) == ["มาตรา 24"]


def test_extract_citations_handles_collapsed_whitespace() -> None:
    # Marker sometimes drops the space between มาตรา and the numeral.
    md = "มาตรา๓๒"
    assert extract_citations(md) == ["มาตรา 32"]


def test_score_page_perfect_match() -> None:
    s = score_page(
        parser="marker",
        page_index=1,
        expected=["มาตรา ๒", "มาตรา ๓"],
        extracted=["มาตรา 2", "มาตรา 3"],
    )
    assert s.recall == 1.0
    assert s.precision == 1.0
    assert s.false_positives == []
    assert s.false_negatives == []


def test_score_page_partial_overlap() -> None:
    s = score_page(
        parser="typhoon",
        page_index=11,
        expected=["มาตรา ๑", "มาตรา ๒", "มาตรา ๓"],
        extracted=["มาตรา 2", "มาตรา 3", "มาตรา 99"],
    )
    # tp = {มาตรา 2, มาตรา 3}; fp = {มาตรา 99}; fn = {มาตรา 1}
    assert s.recall == 2 / 3
    assert s.precision == 2 / 3
    assert s.false_positives == ["มาตรา 99"]
    assert s.false_negatives == ["มาตรา 1"]


def test_score_page_no_expected_yields_none_recall() -> None:
    s = score_page(parser="marker", page_index=42, expected=[], extracted=["มาตรา 96"])
    assert s.recall is None  # no denominator
    assert s.precision == 0.0  # one extracted, none correct


def test_score_page_no_extracted_yields_none_precision() -> None:
    s = score_page(parser="marker", page_index=16, expected=["มาตรา ๓๐"], extracted=[])
    assert s.recall == 0.0
    assert s.precision is None


def test_aggregate_skips_none_in_mean() -> None:
    pages = [
        score_page("marker", 0, ["มาตรา ๒"], ["มาตรา 2"]),  # recall 1, precision 1
        score_page("marker", 1, [], ["มาตรา 99"]),  # recall None, precision 0
        score_page("marker", 2, ["มาตรา ๓"], []),  # recall 0, precision None
    ]
    agg = aggregate("marker", pages)
    assert agg.page_count == 3
    # mean of valid recalls (1, 0) = 0.5
    assert agg.citation_recall_mean is not None
    assert abs(agg.citation_recall_mean - 0.5) < 1e-9
    # mean of valid precisions (1, 0) = 0.5
    assert agg.citation_precision_mean is not None
    assert abs(agg.citation_precision_mean - 0.5) < 1e-9
    # reading-order + structure unfilled by reviewer → None
    assert agg.reading_order_mean is None
    assert agg.structure_preserved_frac is None


def test_aggregate_with_reviewer_fills() -> None:
    pages = [
        score_page("typhoon", 0, ["มาตรา ๒"], ["มาตรา 2"]),
        score_page("typhoon", 1, ["มาตรา ๓"], ["มาตรา 3"]),
    ]
    pages[0].reading_order_1_to_5 = 4
    pages[0].structure_preserved_0_or_1 = 1
    pages[1].reading_order_1_to_5 = 5
    pages[1].structure_preserved_0_or_1 = 0
    agg = aggregate("typhoon", pages)
    assert agg.reading_order_mean == 4.5
    assert agg.structure_preserved_frac == 0.5
