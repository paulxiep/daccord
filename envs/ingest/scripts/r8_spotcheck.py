"""R8 spot-check: compare browser-print PDF parse quality vs regulator-PDF baseline.

Per dev plan §9.6 + risk-register R8: the UK-GDPR, UK DPA 2018, and FR Loi I+L
sources come from browser-print PDFs (legislation.gov.uk + Légifrance expose no
scraper-friendly consolidated PDFs); the rest are regulator-issued. This script
computes per-source-page metrics for every parsed doc and prints a comparison
table so you can decide whether the R8 fallback (Legifrance API / print-CSS
suppression) is needed.

Metrics (per source page in the input PDF):
- `chars_per_page` — markdown char count / pdf page count (text density)
- `citations_per_page` — count of citation-marker regex hits / pdf page count

Baselines (regulator-PDFs): BDSG DE, BDSG EN, GDPR, etc. Browser-print: UK + FR.

Decision rule (from dev plan §9.6):
- If browser-print citation density < 50% of regulator baseline → schedule R8
  fallback as follow-up task.
- Otherwise → document numbers in dev plan §9.6 and continue to tier 5.
"""

from __future__ import annotations

import re
from pathlib import Path

from daccord.ingest.manifest import IngestManifestEntry, read_manifest
from daccord.validation import validated

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "ingest" / "manifest.jsonl"

# Permissive citation-marker regex covering EN ("Article|Section|Art.|Sec."),
# DE ("§ N"), TH ("มาตรา N"), FR ("Article N"). We're not extracting precise IDs
# here (that's tier 5); we just count how many citation-like marks appear, as a
# proxy for whether Marker's text emission preserves the structural markers.
_CITE_RE = re.compile(
    r"(?:§\s*\d+|\bArticle\s+\d+|\bArt\.?\s+\d+|\bSection\s+\d+|\bSec\.?\s+\d+|มาตรา\s*[๐-๙0-9]+)",
    re.IGNORECASE,
)

# Browser-print sources per dev plan R8. Compared as a group vs the rest.
BROWSER_PRINT_KEYS: set[tuple[str, str]] = {
    ("uk_gdpr", "uk_gdpr_current.pdf"),
    ("dpa_2018", "dpa_2018_current.pdf"),
    ("loi_il", "loi_78_17_consolidated.pdf"),
}


@validated
def citation_count(md_path: Path) -> int:
    """Return the number of citation-marker hits in `md_path`."""
    text = md_path.read_text(encoding="utf-8")
    return sum(1 for _ in _CITE_RE.finditer(text))


@validated
def report(manifest_path: Path = DEFAULT_MANIFEST) -> int:
    entries = [e for e in read_manifest(manifest_path) if not e.failed and e.md_relpath]
    if not entries:
        print(f"no successful entries in {manifest_path}")
        return 2

    rows: list[tuple[str, int, int, float, float, bool]] = []
    for e in entries:
        md_abs = REPO_ROOT / e.md_relpath if e.md_relpath else None
        if md_abs is None or not md_abs.exists():
            continue
        n_cites = citation_count(md_abs)
        pages = e.page_count or 1
        chars_per_page = (e.char_count or 0) / pages
        cites_per_page = n_cites / pages
        is_browser_print = (e.framework, Path(e.pdf_relpath).name) in BROWSER_PRINT_KEYS
        rows.append(
            (
                e.pdf_relpath,
                e.page_count or 0,
                n_cites,
                chars_per_page,
                cites_per_page,
                is_browser_print,
            )
        )

    rows.sort(key=lambda r: (r[5], r[0]))  # browser-print last for visual contrast

    header = (
        f"{'pdf_relpath':<58} {'pages':>6} {'cites':>6} {'chars/p':>10} {'cites/p':>10} {'src':>4}"
    )
    print(header)
    print("-" * 100)
    for relpath, pages, cites, cpp, cppage, bp in rows:
        src = "BP" if bp else "REG"
        print(f"{relpath:<58} {pages:>6} {cites:>6} {cpp:>10.1f} {cppage:>10.2f} {src:>4}")

    reg_rows = [r for r in rows if not r[5]]
    bp_rows = [r for r in rows if r[5]]
    if reg_rows:
        reg_cpp = sum(r[3] for r in reg_rows) / len(reg_rows)
        reg_cps = sum(r[4] for r in reg_rows) / len(reg_rows)
    else:
        reg_cpp = reg_cps = 0.0
    if bp_rows:
        bp_cpp = sum(r[3] for r in bp_rows) / len(bp_rows)
        bp_cps = sum(r[4] for r in bp_rows) / len(bp_rows)
    else:
        bp_cpp = bp_cps = 0.0

    print()
    print(f"Regulator avg:     chars/p={reg_cpp:.1f}  cites/p={reg_cps:.2f}  (n={len(reg_rows)})")
    print(f"Browser-print avg: chars/p={bp_cpp:.1f}  cites/p={bp_cps:.2f}  (n={len(bp_rows)})")
    if reg_cps > 0:
        ratio = bp_cps / reg_cps
        verdict = "PASS" if ratio >= 0.5 else "FAIL (schedule R8 fallback)"
        print(f"Browser-print citation density vs regulator: {ratio:.2f}× → {verdict}")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = p.parse_args(argv)
    return report(args.manifest)


if __name__ == "__main__":
    import sys

    sys.exit(main())


# Helper exposed so tests / other callers can compute the per-row metric
# without going through the printer. Not currently used by the script itself,
# but kept on the module surface for future tier-5 reuse.
@validated
def per_doc_metrics(entry: IngestManifestEntry, md_abs: Path) -> tuple[float, float, int]:
    """Return (chars_per_page, cites_per_page, total_cites) for one entry."""
    n_cites = citation_count(md_abs)
    pages = entry.page_count or 1
    return (entry.char_count or 0) / pages, n_cites / pages, n_cites
