"""Mocked tests for daccord.ingest.marker_runner — CI-runnable without a GPU.

Stubs `marker.converters.pdf`, `marker.models`, `marker.output`, and `pymupdf`
via `sys.modules` injection (mirror envs/baseline/tests/test_local_hf_client.py).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def fake_marker(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Inject a fake marker stack returning deterministic markdown text."""
    state: dict[str, Any] = {"convert_calls": 0, "markdown": "# Title\n\nSome body text\n"}

    marker_root = types.ModuleType("marker")
    marker_root.__version__ = "1.10.2-fake"

    class _FakeRendered:
        pass

    class _FakeConverter:
        def __init__(self, artifact_dict: Any) -> None:
            self.artifact_dict = artifact_dict

        def __call__(self, _pdf_path: str) -> _FakeRendered:
            state["convert_calls"] += 1
            return _FakeRendered()

    converters_mod = types.ModuleType("marker.converters")
    converters_pdf_mod = types.ModuleType("marker.converters.pdf")
    converters_pdf_mod.PdfConverter = _FakeConverter  # type: ignore[attr-defined]

    models_mod = types.ModuleType("marker.models")
    models_mod.create_model_dict = lambda: {"sentinel": True}  # type: ignore[attr-defined]

    output_mod = types.ModuleType("marker.output")
    output_mod.text_from_rendered = lambda _r: (state["markdown"], "md", [])  # type: ignore[attr-defined]

    class _FakeDoc:
        def __init__(self, n: int) -> None:
            self.page_count = n

        def __enter__(self) -> _FakeDoc:
            return self

        def __exit__(self, *_a: Any) -> None: ...

    pymupdf_mod = types.ModuleType("pymupdf")
    pymupdf_mod.open = lambda _p: _FakeDoc(7)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "marker", marker_root)
    monkeypatch.setitem(sys.modules, "marker.converters", converters_mod)
    monkeypatch.setitem(sys.modules, "marker.converters.pdf", converters_pdf_mod)
    monkeypatch.setitem(sys.modules, "marker.models", models_mod)
    monkeypatch.setitem(sys.modules, "marker.output", output_mod)
    monkeypatch.setitem(sys.modules, "pymupdf", pymupdf_mod)
    return state


def test_parser_version_returns_marker_version(fake_marker: dict[str, Any]) -> None:
    from daccord.ingest.marker_runner import parser_version

    assert parser_version() == "1.10.2-fake"


def test_parser_version_falls_back_to_importlib_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When `marker.__version__` is missing, fall back to importlib.metadata."""
    import sys
    import types

    marker_mod = types.ModuleType("marker")  # no __version__
    monkeypatch.setitem(sys.modules, "marker", marker_mod)
    monkeypatch.setattr(
        "importlib.metadata.version",
        lambda dist: "9.9.9-meta" if dist == "marker-pdf" else "wrong",
    )

    from daccord.ingest.marker_runner import parser_version

    assert parser_version() == "9.9.9-meta"


def test_make_converter_uses_model_dict(fake_marker: dict[str, Any]) -> None:
    from daccord.ingest.marker_runner import make_converter

    converter = make_converter()
    assert getattr(converter, "artifact_dict", None) == {"sentinel": True}


def test_parse_document_writes_md_and_reports_stats(
    fake_marker: dict[str, Any], tmp_path: Path
) -> None:
    from daccord.ingest.marker_runner import make_converter, parse_document

    pdf = tmp_path / "input.pdf"
    pdf.write_bytes(b"%PDF-fake")
    out = tmp_path / "sub" / "out.md"

    converter = make_converter()
    result = parse_document(pdf, out, converter)

    assert out.exists()
    assert out.read_text(encoding="utf-8") == "# Title\n\nSome body text\n"
    assert result.char_count == len("# Title\n\nSome body text\n")
    assert result.page_count == 7
    assert result.seconds_elapsed >= 0
    assert fake_marker["convert_calls"] == 1


def test_parse_document_reuses_one_converter_across_calls(
    fake_marker: dict[str, Any], tmp_path: Path
) -> None:
    from daccord.ingest.marker_runner import make_converter, parse_document

    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    pdf_a.write_bytes(b"%PDF-a")
    pdf_b.write_bytes(b"%PDF-b")

    converter = make_converter()
    parse_document(pdf_a, tmp_path / "a.md", converter)
    parse_document(pdf_b, tmp_path / "b.md", converter)

    # The model dict is built once (via make_converter); the converter is
    # invoked twice. This is the tier-4 amortisation invariant.
    assert fake_marker["convert_calls"] == 2
