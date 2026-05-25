"""Citation-ID extraction + score aggregation for the Thai parser bake-off.

The PDPA-TH source uses Thai-numeral section identifiers (e.g. ``มาตรา ๙๓``).
Both Marker and Typhoon-OCR may emit either Thai numerals (faithful to source)
or Arabic numerals (post-OCR normalisation). To compare against a single hand-
built gold list we normalise both sides to Arabic numerals before set overlap.
"""

from __future__ import annotations

import re
from typing import Final

from daccord.validation import ValidatedModel, validated

_THAI_TO_ARABIC: Final = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

# Matches a section identifier on the form "มาตรา <digits>" where digits are
# Thai or Arabic. Whitespace between "มาตรา" and the digits is optional/varied
# (Marker often collapses, Typhoon often preserves a regular space, source PDF
# uses a Thai whitespace char). The trailing word-boundary keeps "มาตรา ๙๓" out
# of "มาตรา ๙๓๓" matches.
_MATRA_RE: Final = re.compile(r"มาตรา\s*([๐-๙0-9]+)(?![๐-๙0-9])")


class PageScore(ValidatedModel):
    """Per-(parser, page) scoring row in `score_table.csv`."""

    parser: str
    page_index: int
    expected_citations: list[str]
    extracted_citations: list[str]
    true_positives: list[str]
    false_positives: list[str]
    false_negatives: list[str]
    recall: float | None
    precision: float | None

    # Filled in by the human reviewer post-run. CSV writer emits empty strings
    # for None; reviewer edits the CSV in place to record 1-5 + 0/1.
    reading_order_1_to_5: int | None = None
    structure_preserved_0_or_1: int | None = None
    notes: str = ""


class ParserAggregate(ValidatedModel):
    """Per-parser roll-up across all sample pages — what `summary.md` reports."""

    parser: str
    page_count: int
    citation_recall_mean: float | None
    citation_precision_mean: float | None
    reading_order_mean: float | None
    structure_preserved_frac: float | None


@validated
def normalize_thai_numerals(s: str) -> str:
    """Return `s` with Thai digits ๐-๙ remapped to ASCII 0-9."""
    return s.translate(_THAI_TO_ARABIC)


@validated
def extract_citations(markdown_text: str) -> list[str]:
    """Return the sorted, unique set of `'มาตรา <N>'` citations in `markdown_text`.

    Numbers are normalised to Arabic so equivalent-but-differently-encoded hits
    (e.g. ``มาตรา ๙๓`` vs ``มาตรา 93``) collapse to one canonical citation.
    """
    hits = {
        f"มาตรา {normalize_thai_numerals(m.group(1))}" for m in _MATRA_RE.finditer(markdown_text)
    }
    return sorted(hits)


@validated
def _norm_citations(cites: list[str]) -> list[str]:
    """Apply Thai→Arabic normalisation to each input citation string."""
    return [normalize_thai_numerals(c) for c in cites]


@validated
def score_page(
    parser: str,
    page_index: int,
    expected: list[str],
    extracted: list[str],
) -> PageScore:
    """Compute precision/recall of `extracted` against `expected` citation IDs.

    Both sides are Thai→Arabic normalised before set comparison. Recall is
    `None` when `expected` is empty (no gold citations on this page → no
    denominator to divide by); precision is `None` when `extracted` is empty.
    """
    exp_set = set(_norm_citations(expected))
    got_set = set(_norm_citations(extracted))

    tp = sorted(exp_set & got_set)
    fp = sorted(got_set - exp_set)
    fn = sorted(exp_set - got_set)

    recall = len(tp) / len(exp_set) if exp_set else None
    precision = len(tp) / len(got_set) if got_set else None

    return PageScore(
        parser=parser,
        page_index=page_index,
        expected_citations=sorted(expected),
        extracted_citations=sorted(extracted),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        recall=recall,
        precision=precision,
    )


@validated
def _mean_or_none(xs: list[float]) -> float | None:
    """Arithmetic mean of `xs`; `None` if the list is empty."""
    return sum(xs) / len(xs) if xs else None


@validated
def aggregate(parser: str, scores: list[PageScore]) -> ParserAggregate:
    """Roll up per-page scores into the per-parser line of `summary.md`.

    Reading-order + structure scores come from the human reviewer's CSV edits;
    if either is fully unfilled the corresponding mean is `None` so the CSV/MD
    can render a clear "pending" marker rather than a misleading 0.0.
    """
    recalls = [s.recall for s in scores if s.recall is not None]
    precisions = [s.precision for s in scores if s.precision is not None]
    reading_orders = [
        float(s.reading_order_1_to_5) for s in scores if s.reading_order_1_to_5 is not None
    ]
    structure_marks = [
        s.structure_preserved_0_or_1 for s in scores if s.structure_preserved_0_or_1 is not None
    ]

    return ParserAggregate(
        parser=parser,
        page_count=len(scores),
        citation_recall_mean=_mean_or_none(recalls),
        citation_precision_mean=_mean_or_none(precisions),
        reading_order_mean=_mean_or_none(reading_orders),
        structure_preserved_frac=(
            sum(structure_marks) / len(structure_marks) if structure_marks else None
        ),
    )
