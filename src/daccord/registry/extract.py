"""Tier-5 orchestration: one framework's markdown → one FrameworkRegistry.

Inputs are the parsed Marker outputs in `data/ingest/<jur>/<framework>/*.md`
(produced by tier 4) plus their SHA256s from `data/ingest/manifest.jsonl`.

Output is a `FrameworkRegistry` ready to be written via
`daccord.registry.schema.write_registry`. The CLI in
`scripts/extract_registry.py` is the caller; this module stays I/O-light so
tests can exercise it with fixture markdown strings.

The toy-gold recall computation lives here too (`compute_toy_gold_recall`) —
it is the empirical M1 gate-closing anchor.
"""

from __future__ import annotations

from pathlib import Path

from daccord.eval.scoring import normalize_citation_id
from daccord.gold import GoldSet
from daccord.registry.patterns import FRAMEWORK_EXTRACTORS, base_section, sort_key
from daccord.registry.schema import FrameworkRegistry
from daccord.validation import validated

EXTRACTOR_VERSION = "tier-5/v1"


@validated
def extract_framework(
    framework_id: str,
    jurisdiction: str,
    md_texts: list[str],
    source_documents: list[str],
    source_sha256: list[str],
) -> FrameworkRegistry:
    """Run the framework's extractor over each markdown text, dedupe, build a registry.

    `md_texts`, `source_documents`, and `source_sha256` are parallel lists —
    one entry per input markdown (e.g. PDPA-MY has two: the bilingual Act 709
    plus the 2024 amendment). The deduped union of citation IDs across all
    inputs becomes the registry.

    Raises `KeyError` if `framework_id` has no extractor registered — the
    list in `FRAMEWORK_EXTRACTORS` is authoritative.
    """
    if framework_id not in FRAMEWORK_EXTRACTORS:
        raise KeyError(
            f"no extractor registered for framework_id={framework_id!r}; "
            f"known: {sorted(FRAMEWORK_EXTRACTORS.keys())}"
        )
    if len(md_texts) != len(source_documents) or len(md_texts) != len(source_sha256):
        raise ValueError("md_texts / source_documents / source_sha256 must have the same length")
    extractor = FRAMEWORK_EXTRACTORS[framework_id]
    seen: dict[str, str] = {}
    for md in md_texts:
        for canon, display in extractor(md):
            seen.setdefault(canon, display)
    # Re-sort the merged dict by the same key the per-doc extractor uses so
    # the registry order is deterministic regardless of doc iteration order.
    sorted_pairs = sorted(seen.items(), key=sort_key)
    canon_ids = [p[0] for p in sorted_pairs]
    display_ids = [p[1] for p in sorted_pairs]
    return FrameworkRegistry(
        framework=framework_id,
        jurisdiction=jurisdiction,
        citation_ids=canon_ids,
        display_ids=display_ids,
        source_documents=list(source_documents),
        source_sha256=list(source_sha256),
        extractor_version=EXTRACTOR_VERSION,
        citation_count=len(canon_ids),
    )


@validated
def compute_toy_gold_recall(
    framework_id: str,
    registry_ids: list[str],
    toy_gold_path: Path,
) -> tuple[float, list[str]]:
    """Return (recall, sorted_missing_base_sections) for one framework.

    For every distinct citation_id in `toy_gold_path` (as source OR target)
    that belongs to `framework_id`, normalize via `normalize_citation_id`,
    strip subsection suffix via `base_section`, check membership in
    `registry_ids`. Recall is `len(hits) / len(distinct_gold_bases)`.

    A framework that doesn't appear in the toy gold returns `(1.0, [])` — by
    convention "no gold to fail against" is treated as pass, not as None,
    because the M1 gate condition C only requires recall=1.0 for frameworks
    that appear.
    """
    if not toy_gold_path.exists():
        return (1.0, [])
    gold = GoldSet.from_jsonl(toy_gold_path)
    bases: set[str] = set()
    for pair in gold.pairs:
        if pair.source_framework == framework_id:
            bases.add(base_section(normalize_citation_id(pair.source_citation_id)))
        if pair.target_framework == framework_id:
            bases.add(base_section(normalize_citation_id(pair.target_citation_id)))
    if not bases:
        return (1.0, [])
    canon_set = {base_section(c) for c in registry_ids}
    hits = bases & canon_set
    misses = sorted(bases - canon_set)
    return (len(hits) / len(bases), misses)
