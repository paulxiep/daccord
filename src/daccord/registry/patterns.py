"""Per-framework regex authoring — the language-specific knowledge for tier 5.

Each framework's parsed Marker markdown uses a different section-marker
convention:

  - GDPR / UK-GDPR / FR Loi I+L:  "Article N"
  - PDPA-SG / DPA 2012 PH:        "Section N" / "SECTION N." / "SEC. N."
  - UK DPA 2018:                  Section numbers as bare heading text
                                  (`### **45 Right of access**`) plus body
                                  cross-refs `s. 45`, `S. 45(2)`
  - BDSG-DE:                      "§ N"            (paragraph sign)
  - BDSG-EN:                      "Section N"      (English translation)
  - PDPA-MY (bilingual):          "Seksyen N" (MS) + "Section N" (EN)
  - PDPA-TH-thai:                 "มาตรา N"        (Thai or Arabic digits)
  - PDPA-TH-english:              "Section N"

Letter-suffixes (`26A`, `15A`, `38a`) ARE captured because they are legally
distinct sections (toy gold cites `Section 26D` for PDPA-SG). Subsection
precision (`Article 6(1)(a)`) is NOT enumerated — downstream Tier-6A
validates via prefix-match on the canonical base section.

Each extractor returns `list[tuple[canonical_id, display_id]]`. Canonical IDs
are the output of `daccord.eval.scoring.normalize_citation_id` (M0-locked),
so registry hits and eval Tier-1 hits share key space.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Final

from daccord.bakeoff.scoring import normalize_thai_numerals
from daccord.eval.scoring import normalize_citation_id
from daccord.validation import validated

# ─────────────────────────────────────────────────────────────────────────────
# Regex library. `\d+[A-Za-z]?` allows the trailing letter-suffix convention
# common in amended laws (PDPA-SG 26A/26D, BDSG-DE 38a). The `(?![\w])` /
# `\b` tail keeps "Article 99" out of "Article 999" matches.
# ─────────────────────────────────────────────────────────────────────────────

# Capital A. Singular "Article N" and plural "Articles N" both contribute
# IDs — Marker sometimes drops the "Article N" heading entirely (R8 noise on
# the GDPR consolidated PDF, where Article 14/15 lose their headings between
# Article 13's content and Article 16's heading). Body text cross-refs
# ("Articles 15 to 22") are dense enough to backfill those gaps.
# Lowercase "article" cross-refs in FR body text are intentionally ignored
# (they're frequent and add no new IDs).
_ARTICLE_RE: Final = re.compile(r"\bArticles?\s+(\d+)(?![\d])")

# "Articles N to M" — capture the range endpoints so we can expand to all
# intermediate IDs. Run AFTER `_ARTICLE_RE` so the endpoint singletons get
# their own match too.
_ARTICLE_RANGE_RE: Final = re.compile(r"\bArticles?\s+(\d+)\s+to\s+(\d+)\b")

# "Section N" — capital S, matches PDPA-SG and PH body cross-refs.
_SECTION_RE: Final = re.compile(r"\bSection\s+(\d+[A-Za-z]?)(?![\w])")

# Bullet section heading: PDPA-SG and PDPA-MY render TOC + body section
# headings as bullet items like `- 13. Consent required` or `- 26D. Duty…`.
# The `\S` tail rejects bullets that are just a number alone (rare but
# would be ambiguous).
_BULLET_NUM_RE: Final = re.compile(r"^-\s+\*{0,2}(\d+[A-Za-z]*)\.\s+\S", re.MULTILINE)

# Bold inline section heading: PDPA-SG body has `- **13.** An organisation…`
# and `**26D.**—(1)`. The double-asterisk distinguishes section headings
# from regular numbered bullets in body content.
_BOLD_NUM_RE: Final = re.compile(r"\*\*(\d+[A-Za-z]*)\.\*\*")

# Philippines DPA 2012 uses both "SECTION N." (full word) and "SEC. N." (abbrev).
_SECTION_UPPER_RE: Final = re.compile(r"\bSECTION\s+(\d+[A-Za-z]?)(?![\w])")
_SEC_DOT_RE: Final = re.compile(r"\bSEC\.\s*(\d+[A-Za-z]?)(?![\w])")

# UK DPA 2018: body cross-refs `s. 45` / `S. 45(2)`. Word-boundary on the
# left so "ms. 4" or similar abbreviations don't false-match — the `[Ss]`
# must be preceded by a word boundary.
_S_DOT_RE: Final = re.compile(r"(?<![A-Za-z])[Ss]\.\s+(\d+[A-Za-z]?)(?![\w])")

# UK DPA 2018 headings: `### **45 Title**` — a bold heading whose first token
# is the section number. Constrained by markdown context (after `###` and
# `**`) so we don't catch numbered list items in body text.
_DPA_2018_HEADING_RE: Final = re.compile(r"^#{1,6}\s+\*{1,2}(\d+[A-Za-z]?)\s+", re.MULTILINE)

# BDSG-DE: `§ N` (paragraph sign + optional space + digits + optional
# lowercase letter suffix like "38a"). The non-letter-after constraint
# rejects matches like "§ 38a1".
_PARAGRAPH_RE: Final = re.compile(r"§\s*(\d+[a-z]?)(?![\w])")

# PDPA-MY (Malay): "Seksyen N".
_SEKSYEN_RE: Final = re.compile(r"\bSeksyen\s+(\d+[A-Za-z]?)(?![\w])")

# PDPA-TH (Thai): "มาตรา" + Thai (๐-๙) or Arabic digits. Trailing negative
# lookahead is mixed-digit-aware so "มาตรา ๙๓" out of "มาตรา ๙๓๓" stays clean.
_MATRA_RE: Final = re.compile(r"มาตรา\s*([๐-๙0-9]+[A-Za-z]?)(?![๐-๙0-9])")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — apply a list of regex passes, dedupe, canonicalise, return pairs.
# ─────────────────────────────────────────────────────────────────────────────


@validated
def _canonicalize(display: str) -> str:
    """Run the M0-locked eval normalizer + uppercase letter suffix.

    `normalize_citation_id` lowercases everything; we then re-uppercase the
    trailing letter so the canonical-ID space matches the toy-gold convention
    ("26D" not "26d"). For lowercase-suffix laws (BDSG-DE "38a") this is a
    one-way mapping that doesn't cause collisions within a single framework
    because each framework uses one suffix-case convention.
    """
    canon = normalize_citation_id(display)
    return re.sub(r"([a-z])$", lambda m: m.group(1).upper(), canon)


@validated
def _harvest(md: str, passes: list[tuple[re.Pattern[str], str]]) -> list[tuple[str, str]]:
    """Run each `(regex, display_template)` pass over `md`, dedupe by canonical_id.

    `display_template` is a format string with a single `{n}` placeholder for
    the captured number — e.g. `"Article {n}"`, `"Section {n}"`, `"§ {n}"`.

    Returns pairs sorted ascending by integer section number (so the output
    JSON list reads top-to-bottom like the underlying law).
    """
    seen: dict[str, str] = {}
    for pat, template in passes:
        for m in pat.finditer(md):
            raw = m.group(1)
            display = template.format(n=raw)
            canon = _canonicalize(display)
            if not canon:
                continue
            seen.setdefault(canon, display)
    return sorted(seen.items(), key=sort_key)


def sort_key(pair: tuple[str, str]) -> tuple[int, str]:
    """Sort canonical IDs by numeric prefix, then by full string.

    `("26d", "Section 26D")` → `(26, "26d")` so "5" comes before "26d" comes
    before "65". Falls back to (max_int, str) for anything without a numeric
    prefix.
    """
    canon = pair[0]
    m = re.match(r"^(\d+)", canon)
    return (int(m.group(1)) if m else 10**9, canon)


# ─────────────────────────────────────────────────────────────────────────────
# Per-framework extractors. Dispatched from FRAMEWORK_EXTRACTORS below.
# ─────────────────────────────────────────────────────────────────────────────


@validated
def extract_article_en(md: str) -> list[tuple[str, str]]:
    """GDPR, UK-GDPR, FR Loi I+L: heading + body-text 'Article(s) N' refs.

    Also expands range references like "Articles 15 to 22" to {15..22} so
    Marker's occasional heading drops (Article 14/15 in the GDPR PDF) don't
    create false misses — the dense in-body cross-references backfill them.
    """
    pairs = _harvest(md, [(_ARTICLE_RE, "Article {n}")])
    seen: dict[str, str] = dict(pairs)
    for m in _ARTICLE_RANGE_RE.finditer(md):
        start, end = int(m.group(1)), int(m.group(2))
        if start <= end and end - start <= 100:  # sanity cap on range width
            for n in range(start, end + 1):
                display = f"Article {n}"
                canon = _canonicalize(display)
                if canon:
                    seen.setdefault(canon, display)
    return sorted(seen.items(), key=sort_key)


@validated
def extract_section_en(md: str) -> list[tuple[str, str]]:
    """PDPA-SG: bullet-form `- 13. Title`, bold `**13.**`, plus 'Section N' refs.

    PDPA-SG's actual section headings are bullets — the TOC reads
    `- 13. Consent required` and body section starts are `- **13.** …` or
    `**26D.**—(1)`. The "Section N" pattern catches body cross-refs that
    further densify coverage.
    """
    return _harvest(
        md,
        [
            (_BULLET_NUM_RE, "Section {n}"),
            (_BOLD_NUM_RE, "Section {n}"),
            (_SECTION_RE, "Section {n}"),
        ],
    )


@validated
def extract_dpa_2012_ph(md: str) -> list[tuple[str, str]]:
    """DPA 2012 PH: mixed 'SECTION N.', 'SEC. N.', 'Section N' forms."""
    return _harvest(
        md,
        [
            (_SECTION_RE, "Section {n}"),
            (_SECTION_UPPER_RE, "Section {n}"),
            (_SEC_DOT_RE, "Section {n}"),
        ],
    )


@validated
def extract_dpa_2018(md: str) -> list[tuple[str, str]]:
    """UK DPA 2018: bare-number headings + 's. N' / 'S. N' body cross-refs.

    The browser-print PDF (R8) puts section numbers in bold heading prefix,
    not "Section N" form. We catch those via _DPA_2018_HEADING_RE, then union
    with the body cross-references which densely confirm valid IDs.
    """
    return _harvest(
        md,
        [
            (_DPA_2018_HEADING_RE, "Section {n}"),
            (_S_DOT_RE, "Section {n}"),
            (_SECTION_RE, "Section {n}"),
        ],
    )


@validated
def extract_bdsg(md: str) -> list[tuple[str, str]]:
    """BDSG: § N (DE) + Section N (EN translation), merged."""
    return _harvest(
        md,
        [
            (_PARAGRAPH_RE, "Section {n}"),
            (_SECTION_RE, "Section {n}"),
        ],
    )


@validated
def extract_pdpa_my(md: str) -> list[tuple[str, str]]:
    """PDPA-MY: Seksyen N (MS) + Section N (EN), merged."""
    return _harvest(
        md,
        [
            (_SEKSYEN_RE, "Section {n}"),
            (_SECTION_RE, "Section {n}"),
        ],
    )


@validated
def extract_pdpa_th(md: str) -> list[tuple[str, str]]:
    """PDPA-TH: มาตรา N (TH, Thai or Arabic digits) + Section N (EN), merged.

    Thai-digit IDs are normalized to Arabic in the display form too — so
    `มาตรา ๙๓` and `Section 93` collapse to one entry with display
    `Section 93`. The original Thai form is lost in display, but the registry
    is consumed by Tier 6A as a constraint set, not as a user-facing artifact.
    """
    seen: dict[str, str] = {}
    for m in _MATRA_RE.finditer(md):
        raw = normalize_thai_numerals(m.group(1))
        display = f"Section {raw}"
        canon = _canonicalize(display)
        if canon:
            seen.setdefault(canon, display)
    for m in _SECTION_RE.finditer(md):
        display = f"Section {m.group(1)}"
        canon = _canonicalize(display)
        if canon:
            seen.setdefault(canon, display)
    return sorted(seen.items(), key=sort_key)


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch table — one entry per framework_id in data/sources.yaml.
# ─────────────────────────────────────────────────────────────────────────────

FRAMEWORK_EXTRACTORS: Final[dict[str, Callable[[str], list[tuple[str, str]]]]] = {
    "gdpr": extract_article_en,
    "uk_gdpr": extract_article_en,
    "dpa_2018": extract_dpa_2018,
    "loi_il": extract_article_en,
    "pdpa_sg": extract_section_en,
    "dpa_2012_ph": extract_dpa_2012_ph,
    "bdsg": extract_bdsg,
    "pdpa_my": extract_pdpa_my,
    "pdpa_th": extract_pdpa_th,
}


@validated
def base_section(normalized_citation: str) -> str:
    """Strip subsection suffix from a normalized citation ID.

    `"6(1)(a)"` → `"6"`, `"26D"` → `"26D"`, `"37(4)"` → `"37"`. Returns the
    leading numeric (+ optional letter) prefix; the result is the lookup key
    against `FrameworkRegistry.citation_ids`.
    """
    m = re.match(r"^(\d+[A-Za-z]?)", normalized_citation)
    if not m:
        return normalized_citation
    base = m.group(1)
    return re.sub(r"([a-z])$", lambda x: x.group(1).upper(), base)
