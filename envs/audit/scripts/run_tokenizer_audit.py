"""Tier-2C tokenizer audit CLI — empirically measures how the project's chosen
QLoRA base tokenizer fragments Thai / French / German regulatory text.

Default base (2026-05-25): Qwen3-8B (Qwen/Qwen3-8B). Replaces the original
Qwen2.5-7B-Instruct choice — Qwen 3 has a newer multilingual tokenizer and
similar VRAM footprint at NF4. Pass --model-id to override.

Reads the corpus manifest (data/raw_manifest.json), extracts plain text from
the per-language PDFs via pypdfium2, loads the tokenizer via HuggingFace
(public/ungated — no HF_TOKEN needed; first run downloads tokenizer assets
to the daccord-hf-cache volume), computes metrics via
daccord.tokenizer_audit, writes eval/tokenizer_audit.{md,csv}, and logs
the run to MLflow.

Exit codes:
    0   all languages PASS
    1   any WARN (none FAIL)
    3   any FAIL (M0 or §5)

Run:   (from envs/audit/) uv run python scripts/run_tokenizer_audit.py
Dry:   (from envs/audit/) uv run python scripts/run_tokenizer_audit.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import mlflow
import pypdfium2 as pdfium

from daccord.corpus.manifest import Manifest, ManifestEntry
from daccord.tokenizer_audit import (
    AuditMetrics,
    LanguageSample,
    compute_audit_metrics,
    overall_exit_code,
    render_csv_rows,
    render_markdown,
    verdict_for,
    write_csv,
)
from daccord.tracking import (
    get_git_commit,
    log_standard_params,
    set_all_seeds,
    setup_mlflow,
)
from daccord.validation import validated

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "raw_manifest.json"
DEFAULT_OUT_MD = REPO_ROOT / "eval" / "tokenizer_audit.md"
DEFAULT_OUT_CSV = REPO_ROOT / "eval" / "tokenizer_audit.csv"
DEFAULT_RAW_TEXT_DIR = REPO_ROOT / "eval" / "raw" / "tokenizer_audit"
DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_LANGUAGES = ("th", "fr", "de", "en")
DEFAULT_PAGES = 8
MIN_USEFUL_CHARS = 500

# Language → (framework, filename) lookup against data/raw_manifest.json.
# Thai is the gate-critical sample; EN is the baseline anchor.
LANGUAGE_SOURCES: dict[str, tuple[str, str]] = {
    "th": ("pdpa_th", "pdpa_th_thai_2019.pdf"),
    "fr": ("loi_il", "loi_78_17_consolidated.pdf"),
    "de": ("bdsg", "bdsg_de_current.pdf"),
    "en": ("gdpr", "reg_2016_679_consolidated.pdf"),
}

LANGS_WITHOUT_WORD_BOUNDARIES = frozenset({"th"})

log = logging.getLogger("tokenizer_audit")


@validated
def extract_pdf_text(
    pdf_path: Path,
    pages_to_take: int,
    raw_text_out: Path | None,
) -> tuple[str, tuple[int, int]]:
    """Extract plain text from a midway page band of ``pdf_path``.

    Returns ``(text, (start_page, end_page))`` with the half-open page range
    actually used. If the chosen band yields < ``MIN_USEFUL_CHARS`` characters,
    falls back to the entire document and logs a warning. Optionally persists
    the extracted text to ``raw_text_out`` for reproducibility (gitignored).
    """
    pdf = pdfium.PdfDocument(str(pdf_path))
    n_pages = len(pdf)
    start = max(2, n_pages // 3)
    end = min(start + pages_to_take, n_pages)
    text = _concat_pages(pdf, start, end)
    page_range = (start, end)
    if len(text) < MIN_USEFUL_CHARS:
        log.warning(
            "%s yielded only %d chars on pages [%d,%d) — falling back to whole doc",
            pdf_path.name,
            len(text),
            start,
            end,
        )
        text = _concat_pages(pdf, 0, n_pages)
        page_range = (0, n_pages)
    if raw_text_out is not None:
        raw_text_out.parent.mkdir(parents=True, exist_ok=True)
        raw_text_out.write_text(text, encoding="utf-8")
    return text, page_range


def _concat_pages(pdf: pdfium.PdfDocument, start: int, end: int) -> str:
    parts: list[str] = []
    for i in range(start, end):
        page = pdf[i]
        try:
            textpage = page.get_textpage()
            parts.append(textpage.get_text_bounded())
        finally:
            page.close()
    return "\n".join(parts)


@validated
def audit_one_language(
    lang: str,
    entry: ManifestEntry,
    pages_to_take: int,
    raw_text_dir: Path | None,
    encode_fn,
    decode_fn,
) -> tuple[LanguageSample, AuditMetrics, str]:
    """Extract text, tokenize, compute metrics, and produce a verdict."""
    # Reconstruct the PDF path from manifest fields so it works across host
    # platforms (the manifest's `local_path` field stores the host's absolute
    # path at corpus-download time, which breaks when the audit runs inside
    # the Docker container where /workspace differs from the Windows host root).
    pdf_path = REPO_ROOT / "data" / "raw" / entry.jurisdiction / entry.framework / entry.filename
    raw_out = (raw_text_dir / f"{lang}.txt") if raw_text_dir is not None else None
    text, page_range = extract_pdf_text(pdf_path, pages_to_take, raw_out)
    token_ids: list[int] = encode_fn(text)
    log.info(
        "%s/%s pages=[%d,%d) chars=%d tokens=%d",
        entry.framework,
        entry.filename,
        page_range[0],
        page_range[1],
        len(text),
        len(token_ids),
    )
    metrics = compute_audit_metrics(
        text=text,
        token_ids=token_ids,
        decoder=decode_fn,
        encoder=encode_fn,
        word_split_supported=lang not in LANGS_WITHOUT_WORD_BOUNDARIES,
    )
    sample = LanguageSample(
        lang=lang,
        source_framework=entry.framework,
        source_filename=entry.filename,
        source_sha256=entry.sha256,
        page_range=page_range,
        char_count=len(text),
    )
    verdict = verdict_for(lang, metrics)
    return sample, metrics, verdict


@validated
def resolve_entries(manifest: Manifest, languages: list[str]) -> list[tuple[str, ManifestEntry]]:
    """Look up the (framework, filename) for each language and fail loud if missing."""
    resolved: list[tuple[str, ManifestEntry]] = []
    for lang in languages:
        if lang not in LANGUAGE_SOURCES:
            raise SystemExit(f"unknown language {lang!r}; supported: {sorted(LANGUAGE_SOURCES)}")
        framework, filename = LANGUAGE_SOURCES[lang]
        entry = manifest.find(framework, filename)
        if entry is None:
            raise SystemExit(
                f"manifest missing entry for {framework}/{filename} — "
                f"run scripts/download_corpus.py first"
            )
        resolved.append((lang, entry))
    return resolved


@validated
def run(
    manifest_path: Path,
    out_md: Path,
    out_csv: Path,
    raw_text_dir: Path,
    model_id: str,
    languages: list[str],
    pages_per_lang: int,
    seed: int,
    dry_run: bool,
) -> int:
    manifest = Manifest.load(manifest_path)
    resolved = resolve_entries(manifest, languages)

    log.info("loading tokenizer %s (first run downloads ~10 MB)", model_id)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def encode_fn(s: str) -> list[int]:
        return tokenizer.encode(s, add_special_tokens=False)

    def decode_fn(tid: int) -> str:
        return tokenizer.decode([tid])

    if dry_run:
        log.info("dry-run: tokenizer + manifest resolved; skipping extraction")
        for lang, entry in resolved:
            log.info("  %s -> %s", lang, entry.local_path)
        return 0

    setup_mlflow(experiment_name="daccord-tokenizer-audit")
    set_all_seeds(seed)

    git_commit = get_git_commit()
    with mlflow.start_run(run_name=f"audit-{git_commit}") as run_ctx:
        log_standard_params(
            run_name=f"audit-{git_commit}",
            seed=seed,
            dataset_hash=None,
            extra={"phase": "2C-tokenizer", "model_id": model_id},
        )

        rows = [
            audit_one_language(
                lang=lang,
                entry=entry,
                pages_to_take=pages_per_lang,
                raw_text_dir=raw_text_dir,
                encode_fn=encode_fn,
                decode_fn=decode_fn,
            )
            for lang, entry in resolved
        ]

        for sample, metrics, verdict in rows:
            mlflow.log_metric(f"tokens_per_char__{sample.lang}", metrics.tokens_per_char)
            mlflow.log_metric(
                f"single_byte_frac__{sample.lang}", metrics.single_byte_token_fraction
            )
            mlflow.log_param(f"src_sha256__{sample.lang}", sample.source_sha256)
            mlflow.log_param(
                f"pages__{sample.lang}",
                f"[{sample.page_range[0]},{sample.page_range[1]})",
            )
            mlflow.log_param(f"verdict__{sample.lang}", verdict)

        md = render_markdown(
            rows=rows,
            model_id=model_id,
            git_commit=git_commit,
            run_id=run_ctx.info.run_id,
        )
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md, encoding="utf-8")

        csv_rows = render_csv_rows(rows)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            write_csv(csv_rows, f)

        mlflow.log_artifact(str(out_md))
        mlflow.log_artifact(str(out_csv))

    verdicts = [v for _, _, v in rows]
    log.info("verdicts: %s", dict(zip([s.lang for s, _, _ in rows], verdicts, strict=True)))
    log.info("artifact: %s", out_md)
    return overall_exit_code(verdicts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--raw-text-dir", type=Path, default=DEFAULT_RAW_TEXT_DIR)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--languages",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=list(DEFAULT_LANGUAGES),
        help=f"comma-separated language codes; default {','.join(DEFAULT_LANGUAGES)}",
    )
    parser.add_argument("--pages-per-lang", type=int, default=DEFAULT_PAGES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return run(
        manifest_path=args.manifest,
        out_md=args.out_md,
        out_csv=args.out_csv,
        raw_text_dir=args.raw_text_dir,
        model_id=args.model_id,
        languages=args.languages,
        pages_per_lang=args.pages_per_lang,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
