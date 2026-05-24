"""Tier-2D parser bake-off: Marker vs Typhoon-OCR on a 5-page Thai sample.

The package is import-safe without the heavy parser deps installed; the runner
modules lazy-import marker / transformers / typhoon_ocr so unit tests for the
scoring layer can run in the slim default environment.
"""

from daccord.bakeoff.scoring import (
    PageScore,
    aggregate,
    extract_citations,
    normalize_thai_numerals,
    score_page,
)

__all__ = [
    "PageScore",
    "aggregate",
    "extract_citations",
    "normalize_thai_numerals",
    "score_page",
]
