"""Tier-7A clause-body extractor — citation_id → prose body text.

Companion of `daccord.registry.extract` (which produces citation IDs only).
Tier 7A's ensemble prompt requires the actual clause text (`source_mechanism`)
for each source citation; this module supplies that text by re-parsing the
same `data/ingest/<jur>/<framework>/*.md` files tier 5 already consumed.

Strategy: per-framework "heading anchor" regex. The anchor matches the
on-disk **opening** of a section (markdown heading + bold prefix + section
opener like `Article N` / `§ N` / `**N.**` / `มาตรา N`) — strictly enough to
avoid body cross-references. Anchor matches in document order give section
start offsets; the body of section i is the markdown slice from `match[i].end`
to `match[i+1].start` (or EOF for the last).

When the **same canonical citation_id** appears as multiple anchor matches
(rare; e.g. PDPA-MY's bilingual Act 709 has both `Section N` + `Seksyen N`
headings for the same N), the first occurrence's body wins — both bodies are
the same content in different languages, and the first is the canonical one.

Body recall < 1.0 is expected: a section ID may appear only as a body
cross-reference (no heading), or Marker may have dropped a heading entirely
(R8 noise on the GDPR consolidated PDF). The extractor returns
`FrameworkClauses.missing_citation_ids` listing registry IDs that did NOT
get a body; downstream tier 7A either skips those clauses or uses citation_id
alone as `source_mechanism` (degraded but not broken).

Caveats:
  - PDPA-TH-thai is the lowest-recall case: Marker collapsed many sections
    onto single lines, the heading anchor is `มาตรา N` without markdown
    structure, and the preamble references future sections as cross-refs
    before they're defined. We trim leading matches that occur before the
    first `มาตรา ๑` to drop preamble noise.
  - PDPA-MY's bilingual Act 709 has BM headings (`Seksyen N`) and EN headings
    (`Section N`) for the same section; we prefer the EN body (downstream
    LLMs handle English better and the canonical_id pipeline already collapses
    both into one ID).
  - Section IDs with letter suffixes (`26D`, `38a`) are preserved end-to-end.

Output: `FrameworkClauses` (see `daccord.registry.schema`), persisted at
`data/clauses/{framework}.json` by `scripts/extract_clauses.py`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Final

from daccord.bakeoff.scoring import normalize_thai_numerals
from daccord.eval.scoring import normalize_citation_id
from daccord.registry.schema import FrameworkClauses
from daccord.validation import validated

EXTRACTOR_VERSION = "tier-7A/v1"


# ─────────────────────────────────────────────────────────────────────────────
# Per-framework heading anchors. Each regex captures the citation number in
# group 1; alternation across multiple heading-style variants is allowed (e.g.
# BDSG's `§ N` DE heading + `Section N` EN heading both anchor a section).
#
# `(?m)` enables multiline mode so `^` matches line start. Heading anchors are
# strictly line-start to avoid body cross-references (which usually appear
# mid-line). Where the framework's parsed markdown drops heading prefixes
# (PDPA-SG body uses `**N.**` without `#`, BDSG-EN uses `**Section N`), we
# rely on the bold delimiter `**` as the only practical anchor.
# ─────────────────────────────────────────────────────────────────────────────

# Each heading anchor consumes the FULL heading region (trailing markdown
# decoration, optional subtitle, closing `*` / `**`, newline) so `m.end()`
# lands at the first byte of body content. This keeps leftover heading bits
# (the closing `*` of `*Article 1*`, the subtitle of `**Section 15 Title**`)
# out of the body slice.

# GDPR / UK-GDPR / FR Loi I+L — markdown heading line `#### *Article N*` or
# variants. Consume the entire heading line including trailing newline.
_HEADING_ARTICLE: Final = re.compile(
    r"(?m)^#{1,6}[^\n]*?\bArticle\s+(\d+)(?![\d])[^\n]*\n",
)

# UK DPA 2018 — `### **45 Right of access**`. Bold heading with bare number.
# Consume the whole heading line.
_HEADING_DPA_2018: Final = re.compile(
    r"(?m)^#{1,6}\s+\*{1,2}(\d+[A-Za-z]?)\s+[^\n]*\n",
)

# BDSG — two alternatives:
#   1. `#### **§ 4 Title**` (DE) — markdown heading line, consume whole line.
#   2. `**Section 4 Title**` (EN) — inline bold heading followed by body on
#      the SAME line (Marker's BDSG-EN output collapses headings into bodies);
#      consume the `**Section N Title**` prefix + trailing space.
_HEADING_BDSG: Final = re.compile(
    r"(?m)^(?:#{1,6}[^\n]*?§\s*(\d+[a-z]?)[^\n]*\n|"
    r"\*\*Section\s+(\d+[a-z]?)\s+[^*\n]*\*\*\s*)",
)

# PDPA-MY — `## **Seksyen N. Title**` (BM) and `## **Section N. Title**` (EN).
# Markdown heading line; consume whole line.
_HEADING_PDPA_MY: Final = re.compile(
    r"(?m)^#{1,6}[^\n]*?\b(?:Seksyen|Section)\s+(\d+[A-Za-z]?)\b[^\n]*\n",
)

# PDPA-SG — `**N.**` body-section opener (no markdown heading; just bold).
# Body follows immediately after the closing `**` (sometimes preceded by
# `—(1)` subsection numbering); m.end() lands at the first byte of body.
_HEADING_PDPA_SG: Final = re.compile(
    r"(?m)^\*\*(\d+[A-Za-z]?)\.\*\*",
)

# DPA 2012 PH — `SECTION N.` / `SEC. N.` / `- SEC. N.` at line start, body
# follows on the same line (no markdown heading). Keep the match tight so
# m.end() leaves `*Title*. –` etc. as part of the body — those are useful
# context for the prompt.
_HEADING_DPA_2012_PH: Final = re.compile(
    r"(?m)^(?:-\s+)?(?:SECTION|SEC\.)\s+(\d+[A-Za-z]?)\b",
)

# PDPA-TH-EN — `**Section N**` (inline bold) or `### **Section N**` (markdown
# heading). Consume to closing `**` + trailing whitespace; body follows.
_HEADING_PDPA_TH_EN: Final = re.compile(
    r"(?m)^(?:#{1,6}\s+)?\*\*Section\s*(\d+[A-Za-z]?)[^*\n]*\*\*\s*",
)

# PDPA-TH-thai — `มาตรา N` inline (Marker collapses multiple sections onto
# one line; can't use `^` anchor). Body follows immediately on the same line.
_HEADING_PDPA_TH_THAI: Final = re.compile(
    r"มาตรา\s*([๐-๙0-9]+[A-Za-z]?)(?![๐-๙0-9])",
)


# ─────────────────────────────────────────────────────────────────────────────
# Body extraction — slice markdown between consecutive heading anchors.
# ─────────────────────────────────────────────────────────────────────────────


@validated
def _canonical_id(raw_number: str, display_template: str) -> str:
    """Run M0-locked normalize + restore uppercase letter suffix.

    Mirrors `daccord.registry.patterns._canonicalize` — must stay byte-identical
    so this module's keys hit the same canonical space as the registry's.
    """
    canon = normalize_citation_id(display_template.format(n=raw_number))
    return re.sub(r"([a-z])$", lambda m: m.group(1).upper(), canon)


def _slice_bodies(
    md: str,
    anchor_re: re.Pattern[str],
    display_template: str,
    *,
    normalize_number: Callable[[str], str] = lambda n: n,
) -> dict[str, str]:
    """Generic slicer: find anchor matches, slice body between them.

    Returns `{canonical_id: body_text}` with first-occurrence-wins semantics.
    The `normalize_number` hook handles Thai-digit → Arabic conversion for
    PDPA-TH-thai (other frameworks pass through unchanged).
    """
    matches = list(anchor_re.finditer(md))
    if not matches:
        return {}
    out: dict[str, str] = {}
    for i, m in enumerate(matches):
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        # The anchor regex may have multiple capture groups (BDSG has 3 — one
        # per alternative); take the first non-None group.
        raw_groups = [g for g in m.groups() if g is not None]
        if not raw_groups:
            continue
        raw = normalize_number(raw_groups[0])
        canon = _canonical_id(raw, display_template)
        if not canon:
            continue
        body = md[body_start:body_end].strip()
        if not body:
            continue
        # First occurrence wins — TOC-like duplicate headings (rare but
        # possible) shouldn't overwrite the real body.
        out.setdefault(canon, body)
    return out


def _extract_pdpa_th_thai_clauses(md: str) -> dict[str, str]:
    """PDPA-TH-thai special case: drop preamble cross-refs before มาตรา ๑.

    The Thai parsed markdown opens with preamble paragraphs that reference
    later sections (`มาตรา ๒๖ ประกอบกับมาตรา ๓๒...`) before any section is
    actually defined. The real document body starts at `มาตรา ๑`. Anything
    before the first `มาตรา ๑` (Thai or Arabic form) is preamble noise.
    """
    first_section_re = re.compile(r"มาตรา\s*[๑1]\b")
    first_match = first_section_re.search(md)
    if first_match is None:
        return {}
    body_md = md[first_match.start() :]
    return _slice_bodies(
        body_md,
        _HEADING_PDPA_TH_THAI,
        "Section {n}",
        normalize_number=normalize_thai_numerals,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-framework dispatch — one extractor function per framework_id.
# ─────────────────────────────────────────────────────────────────────────────


@validated
def extract_clauses_gdpr_like(md: str) -> dict[str, str]:
    """GDPR / UK-GDPR / FR Loi I+L: `Article N` markdown headings."""
    return _slice_bodies(md, _HEADING_ARTICLE, "Article {n}")


@validated
def extract_clauses_dpa_2018(md: str) -> dict[str, str]:
    """UK DPA 2018: `### **45 Title**` bare-number bold headings."""
    return _slice_bodies(md, _HEADING_DPA_2018, "Section {n}")


@validated
def extract_clauses_bdsg(md: str) -> dict[str, str]:
    """BDSG: `§ N` (DE markdown heading) and `**Section N` (EN bold-only)."""
    return _slice_bodies(md, _HEADING_BDSG, "Section {n}")


@validated
def extract_clauses_pdpa_my(md: str) -> dict[str, str]:
    """PDPA-MY: bilingual `Seksyen N` (BM) + `Section N` (EN) headings."""
    return _slice_bodies(md, _HEADING_PDPA_MY, "Section {n}")


@validated
def extract_clauses_pdpa_sg(md: str) -> dict[str, str]:
    """PDPA-SG: line-start `**N.**` bold-numbered section bodies."""
    return _slice_bodies(md, _HEADING_PDPA_SG, "Section {n}")


@validated
def extract_clauses_dpa_2012_ph(md: str) -> dict[str, str]:
    """DPA 2012 PH: line-start `SECTION N.` / `SEC. N.` headings."""
    return _slice_bodies(md, _HEADING_DPA_2012_PH, "Section {n}")


@validated
def extract_clauses_pdpa_th(md: str) -> dict[str, str]:
    """PDPA-TH: union of Thai `มาตรา N` + English `**Section N**`.

    Each language's parsed file is passed separately to the caller, so this
    extractor runs twice (once per md_text). The aggregator in
    `extract_framework_clauses` merges results — EN wins on duplicate IDs
    because the EN body is more useful for downstream LLM prompts.
    """
    # Try Thai first; if the input doesn't contain มาตรา, fall back to EN.
    thai_clauses = _extract_pdpa_th_thai_clauses(md)
    if thai_clauses:
        return thai_clauses
    return _slice_bodies(md, _HEADING_PDPA_TH_EN, "Section {n}")


def _missing_sort_key(canon: str) -> tuple[int, str]:
    """Same numeric-prefix-then-string ordering as `patterns.sort_key`."""
    m = re.match(r"^(\d+)", canon)
    return (int(m.group(1)) if m else 10**9, canon)


FRAMEWORK_CLAUSE_EXTRACTORS: Final[dict[str, Callable[[str], dict[str, str]]]] = {
    "gdpr": extract_clauses_gdpr_like,
    "uk_gdpr": extract_clauses_gdpr_like,
    "loi_il": extract_clauses_gdpr_like,
    "dpa_2018": extract_clauses_dpa_2018,
    "bdsg": extract_clauses_bdsg,
    "pdpa_my": extract_clauses_pdpa_my,
    "pdpa_sg": extract_clauses_pdpa_sg,
    "dpa_2012_ph": extract_clauses_dpa_2012_ph,
    "pdpa_th": extract_clauses_pdpa_th,
}


# ─────────────────────────────────────────────────────────────────────────────
# Aggregator — merge across multiple markdown sources + compute body recall.
# ─────────────────────────────────────────────────────────────────────────────


@validated
def extract_framework_clauses(
    framework_id: str,
    jurisdiction: str,
    md_texts: list[str],
    source_documents: list[str],
    source_sha256: list[str],
    registry_citation_ids: list[str],
) -> FrameworkClauses:
    """Run the framework's clause extractor over each markdown text, merge.

    Parallel `md_texts` / `source_documents` / `source_sha256` lists (one
    entry per parsed `.md`). The dedupe rule is first-occurrence-wins across
    inputs in list order, EXCEPT for PDPA-TH where the EN markdown should
    override the Thai one for any IDs that appear in both — EN is more
    useful for LLM consumption.

    `body_recall` = len(clauses) / len(registry_citation_ids). Missing IDs
    (in registry but not extracted) are listed in `missing_citation_ids`
    sorted by `daccord.registry.patterns.sort_key` for stable diffs.
    """
    if framework_id not in FRAMEWORK_CLAUSE_EXTRACTORS:
        raise KeyError(
            f"no clause extractor registered for framework_id={framework_id!r}; "
            f"known: {sorted(FRAMEWORK_CLAUSE_EXTRACTORS.keys())}"
        )
    if len(md_texts) != len(source_documents) or len(md_texts) != len(source_sha256):
        raise ValueError("md_texts / source_documents / source_sha256 must have the same length")
    extractor = FRAMEWORK_CLAUSE_EXTRACTORS[framework_id]
    merged: dict[str, str] = {}
    # PDPA-TH: process EN markdown LAST so it overwrites Thai bodies (we want
    # English text for downstream prompts). Detect by filename heuristic.
    ordered_indices = list(range(len(md_texts)))
    if framework_id == "pdpa_th":
        ordered_indices.sort(key=lambda i: "english" in source_documents[i].lower())
    for i in ordered_indices:
        per_doc = extractor(md_texts[i])
        for canon_id, body in per_doc.items():
            # First-occurrence-wins normally; for PDPA-TH the ordered_indices
            # above puts EN last, so EN overrides Thai by overwriting.
            if framework_id == "pdpa_th":
                merged[canon_id] = body
            else:
                merged.setdefault(canon_id, body)

    # Filter to registry IDs only — extractor noise (cross-refs that slipped
    # past the heading anchor) gets discarded here, surfaced only via
    # `body_recall < 1.0` if many registry IDs went unmatched.
    clauses = {cid: merged[cid] for cid in registry_citation_ids if cid in merged}
    missing = sorted(
        (cid for cid in registry_citation_ids if cid not in clauses),
        key=_missing_sort_key,
    )
    body_recall = len(clauses) / len(registry_citation_ids) if registry_citation_ids else 0.0

    return FrameworkClauses(
        framework=framework_id,
        jurisdiction=jurisdiction,
        clauses=clauses,
        body_recall=body_recall,
        missing_citation_ids=missing,
        source_documents=list(source_documents),
        source_sha256=list(source_sha256),
        extractor_version=EXTRACTOR_VERSION,
    )
