"""Marker adapter for tier-4 full-corpus parse.

Marker loads a heavy layout/OCR model dict per `PdfConverter` instance.
`make_converter()` builds one; `parse_document()` reuses it across many
documents, so a 13-document sweep pays the model-load cost once.

Separate from `daccord.bakeoff.marker_runner` (which works on per-page PDFs
for the 2D bake-off) — that path is preserved verbatim for the bake-off
artifact; this module is the tier-4 production path.
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any

from daccord.validation import ValidatedModel, validated


class DocumentOutput(ValidatedModel):
    pdf_path: Path
    md_path: Path
    markdown: str
    char_count: int
    page_count: int
    seconds_elapsed: float


@validated
def parser_version() -> str:
    """Return the installed `marker-pdf` version, for MLflow params + manifest.

    Marker doesn't expose `__version__` on its top-level module, so fall back to
    `importlib.metadata.version("marker-pdf")` (the distribution name on PyPI).
    """
    marker = importlib.import_module("marker")
    if hasattr(marker, "__version__"):
        return str(marker.__version__)
    try:
        from importlib.metadata import version

        return version("marker-pdf")
    except Exception:
        return "unknown"


@validated
def make_converter() -> Any:
    """Build a `PdfConverter` with the default Marker model dict.

    Heavy — call once per process and reuse across `parse_document` calls.
    Returned converter is opaque to the caller; pass it back into
    `parse_document(..., converter=...)`.
    """
    converters = importlib.import_module("marker.converters.pdf")
    models = importlib.import_module("marker.models")
    return converters.PdfConverter(artifact_dict=models.create_model_dict())


@validated
def _count_pages(pdf_path: Path) -> int:
    """Return the page count of `pdf_path` via pymupdf (cheap, no model load)."""
    fitz = importlib.import_module("pymupdf")
    with fitz.open(str(pdf_path)) as doc:
        return doc.page_count


@validated
def parse_document(
    pdf_path: Path,
    out_md_path: Path,
    converter: Any,
) -> DocumentOutput:
    """Parse `pdf_path` to markdown via Marker, write `out_md_path`, return stats.

    `converter` must be a Marker `PdfConverter` from `make_converter()`. The
    caller owns its lifecycle; passing a pre-built converter is how the tier-4
    sweep amortises the model-load cost across all 13 PDFs.

    Marker's `text_from_rendered` returns `(text, ext, images)`; tier 4 keeps
    only the markdown text (citations are textual and figure crops are not
    consumed downstream).
    """
    output_mod = importlib.import_module("marker.output")

    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rendered = converter(str(pdf_path))
    text, _ext, _images = output_mod.text_from_rendered(rendered)
    elapsed = time.perf_counter() - t0

    out_md_path.write_text(text, encoding="utf-8")
    return DocumentOutput(
        pdf_path=pdf_path,
        md_path=out_md_path,
        markdown=text,
        char_count=len(text),
        page_count=_count_pages(pdf_path),
        seconds_elapsed=elapsed,
    )
