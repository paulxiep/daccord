"""Marker adapter for the parser bake-off.

Marker (`marker-pdf>=1.10`) loads a layout/OCR model dict per `PdfConverter`
instance — heavy. We instantiate it once and convert each per-page PDF in turn,
so the bake-off only pays the model-load cost on the first page.
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path

from daccord.validation import ValidatedModel, validated


class MarkerPageOutput(ValidatedModel):
    page_index: int
    md_path: Path
    markdown: str
    seconds_elapsed: float


@validated
def parser_version() -> str:
    """Return the installed `marker-pdf` version, for MLflow params."""
    marker = importlib.import_module("marker")
    return getattr(marker, "__version__", "unknown")


@validated
def parse_pages(per_page_pdfs: list[Path], out_dir: Path) -> list[MarkerPageOutput]:
    """Run Marker on each single-page PDF, writing `<out_dir>/page_<n>.md`.

    The PDF filename is expected to encode the page index as `page_{n}.pdf`
    (produced by `bakeoff.sample.rasterize_pages`). Markdown text is also
    returned in-memory so the CLI can score without re-reading the files.
    """
    converters = importlib.import_module("marker.converters.pdf")
    models = importlib.import_module("marker.models")
    output_mod = importlib.import_module("marker.output")

    out_dir.mkdir(parents=True, exist_ok=True)
    converter = converters.PdfConverter(artifact_dict=models.create_model_dict())

    results: list[MarkerPageOutput] = []
    for pdf_path in per_page_pdfs:
        page_idx = _page_index_from_stem(pdf_path.stem)
        t0 = time.perf_counter()
        rendered = converter(str(pdf_path))
        text, _, _ = output_mod.text_from_rendered(rendered)
        elapsed = time.perf_counter() - t0

        md_path = out_dir / f"page_{page_idx}.md"
        md_path.write_text(text, encoding="utf-8")
        results.append(
            MarkerPageOutput(
                page_index=page_idx, md_path=md_path, markdown=text, seconds_elapsed=elapsed
            )
        )
    return results


@validated
def _page_index_from_stem(stem: str) -> int:
    """Extract the page index from `page_<n>` filenames."""
    if not stem.startswith("page_"):
        raise ValueError(f"unexpected per-page PDF stem: {stem!r}")
    return int(stem.removeprefix("page_"))
