"""Sample-page extraction for the parser bake-off.

Given the PDPA-TH Thai source PDF and a small list of 0-indexed pages, this
module writes:

  - one single-page PDF per page  → `<out_dir>/page_images/page_{n}.pdf`
  - one rasterised PNG per page   → `<out_dir>/page_images/page_{n}.png`

Marker consumes the per-page PDFs (it expects a PDF on disk); Typhoon-OCR
consumes the PNGs (vision-LM input). Splitting the source PDF up-front keeps
both runners stateless and lets the bake-off skip parsers independently.
"""

from __future__ import annotations

from pathlib import Path

from daccord.validation import ValidatedModel, validated


class PageArtifact(ValidatedModel):
    """One sampled page rendered to both a single-page PDF and a PNG."""

    page_index: int  # 0-indexed in the source PDF
    pdf_path: Path
    png_path: Path


@validated
def rasterize_pages(
    source_pdf: Path,
    page_indices: list[int],
    out_dir: Path,
    dpi: int = 300,
) -> list[PageArtifact]:
    """Extract `page_indices` from `source_pdf` to per-page PDFs + PNGs.

    `dpi=300` is the same setting invoice-parse uses for PaddleOCR rasterisation
    — high enough that Thai diacritics survive but not so high that the PNGs
    balloon past Typhoon's 1800-px ingest cap (the Typhoon runner resizes down).

    Returns the artifact descriptors in input order. Idempotent: re-running
    overwrites existing per-page files (cheap; ~1MB each).
    """
    pymupdf = __import__("pymupdf")
    page_images = out_dir / "page_images"
    page_images.mkdir(parents=True, exist_ok=True)

    src = pymupdf.open(str(source_pdf))
    try:
        artifacts: list[PageArtifact] = []
        zoom = dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)

        for idx in page_indices:
            if not (0 <= idx < len(src)):
                raise ValueError(
                    f"page index {idx} out of range for {source_pdf.name} (len={len(src)})"
                )

            pdf_path = page_images / f"page_{idx}.pdf"
            png_path = page_images / f"page_{idx}.png"

            # Single-page PDF for Marker
            single = pymupdf.open()
            single.insert_pdf(src, from_page=idx, to_page=idx)
            single.save(str(pdf_path), garbage=4, deflate=True)
            single.close()

            # Rasterised PNG for Typhoon (and human spot-check)
            pixmap = src[idx].get_pixmap(matrix=matrix, alpha=False)
            pixmap.save(str(png_path))

            artifacts.append(PageArtifact(page_index=idx, pdf_path=pdf_path, png_path=png_path))
        return artifacts
    finally:
        src.close()
