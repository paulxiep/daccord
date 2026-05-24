"""Tier-2C tokenizer audit — measures how Qwen2.5-7B-Instruct fragments
Thai, French, and German regulatory text before the project commits to it as
the QLoRA base model.

Pure library. Tokenizer + PDF I/O live in the CLI under [envs/audit/scripts/run_tokenizer_audit.py].
Functions here take already-extracted text + already-encoded token ids + a
``decoder`` callable, so tests pass synthetic data with no HF network access.

Gating thresholds (from [docs/development_plan.md]):

- Thai ``tokens_per_char`` >= 2.0 → ``FAIL_M0`` (M0 §4 cut criterion: swap
  base to SeaLLM-v3 / Typhoon-7B, or descope Thai)
- Thai ``single_byte_token_fraction`` >= 0.20 → ``FAIL_S5`` (§5 hard stop)
- Thai ``tokens_per_char`` >= 2.5 OR ``single_byte_token_fraction`` >= 0.15
  → ``WARN_R4`` (R4 early warning)
- FR/DE/EN: ``WARN`` if ``tokens_per_char`` > 1.5 (Latin-script regression
  sanity check); else ``PASS``.

"Byte-fallback" for Qwen2.5's byte-level BPE (BBPE) is operationally defined
as the fraction of emitted tokens whose decoded UTF-8 form is exactly one
byte — i.e. BPE leaves that never merged into a multi-byte token. This is the
actionable proxy for §5's ">20% bytefallback on Thai = hard stop".
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from daccord.validation import ValidatedModel, validated


class LanguageSample(ValidatedModel):
    """One source-document sample for one language under audit."""

    lang: str
    source_framework: str
    source_filename: str
    source_sha256: str
    page_range: tuple[int, int]
    char_count: int


class AuditMetrics(ValidatedModel):
    """Tokenization metrics computed for one language sample."""

    total_chars: int
    total_tokens: int
    tokens_per_char: float
    tokens_per_word: float | None
    single_byte_token_fraction: float
    token_char_len_p50: float
    token_char_len_p90: float
    token_char_len_p99: float
    top_fragmented_chars: list[tuple[str, int]]


VERDICT_PASS = "PASS"
VERDICT_WARN_LATIN = "WARN"
VERDICT_WARN_R4 = "WARN_R4"
VERDICT_FAIL_M0 = "FAIL_M0"
VERDICT_FAIL_S5 = "FAIL_S5"

_LATIN_LANGS = frozenset({"fr", "de", "en"})

# Thresholds — single source of truth. Re-exported for tests.
THAI_FAIL_M0_TOKENS_PER_CHAR = 2.0
THAI_FAIL_S5_SINGLE_BYTE_FRAC = 0.20
THAI_WARN_R4_TOKENS_PER_CHAR = 2.5
THAI_WARN_R4_SINGLE_BYTE_FRAC = 0.15
LATIN_WARN_TOKENS_PER_CHAR = 1.5


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (p / 100.0) * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


@validated
def compute_audit_metrics(
    text: str,
    token_ids: list[int],
    decoder: Callable[[int], str],
    encoder: Callable[[str], list[int]],
    word_split_supported: bool = True,
    top_fragmented_n: int = 20,
) -> AuditMetrics:
    """Compute tokenization metrics for one language sample.

    Args:
        text: the raw source text the tokenizer was applied to.
        token_ids: the encoded token id sequence
            (``tokenizer.encode(text, add_special_tokens=False)``).
        decoder: ``lambda tid: tokenizer.decode([tid])`` — injected so unit
            tests can pass a deterministic stub without loading HF. Used for
            the single-byte-token fraction.
        encoder: ``lambda s: tokenizer.encode(s, add_special_tokens=False)`` —
            injected; used to compute per-char fragmentation by encoding each
            unique non-ASCII codepoint *alone* (the cleanest BBPE diagnostic;
            decoded-piece counting silently misses split-byte cases because
            partial bytes don't decode back to the original char).
        word_split_supported: ``False`` for scripts without whitespace word
            boundaries (Thai); causes ``tokens_per_word`` to be reported as
            ``None`` instead of polluted by whitespace-split nonsense.
        top_fragmented_n: how many worst-fragmented chars to surface.
    """
    total_chars = len(text)
    total_tokens = len(token_ids)
    tokens_per_char = (total_tokens / total_chars) if total_chars else 0.0

    if word_split_supported:
        word_count = len(text.split())
        tokens_per_word = (total_tokens / word_count) if word_count else 0.0
    else:
        tokens_per_word = None

    decoded_pieces = [decoder(tid) for tid in token_ids]
    single_byte_count = sum(
        1 for piece in decoded_pieces if len(piece.encode("utf-8", errors="replace")) == 1
    )
    single_byte_token_fraction = (single_byte_count / total_tokens) if total_tokens else 0.0

    char_lens = sorted(float(len(piece)) for piece in decoded_pieces)
    p50 = _percentile(char_lens, 50.0)
    p90 = _percentile(char_lens, 90.0)
    p99 = _percentile(char_lens, 99.0)

    top_fragmented = _top_fragmented_chars(text, encoder, top_fragmented_n)

    return AuditMetrics(
        total_chars=total_chars,
        total_tokens=total_tokens,
        tokens_per_char=tokens_per_char,
        tokens_per_word=tokens_per_word,
        single_byte_token_fraction=single_byte_token_fraction,
        token_char_len_p50=p50,
        token_char_len_p90=p90,
        token_char_len_p99=p99,
        top_fragmented_chars=top_fragmented,
    )


def _top_fragmented_chars(
    text: str,
    encoder: Callable[[str], list[int]],
    top_n: int,
) -> list[tuple[str, int]]:
    """For each unique non-ASCII printable char in ``text``, count tokens needed
    to encode it standalone. Return the worst ``top_n``, sorted descending.

    Standalone-encoding is the cleanest BBPE fragmentation diagnostic: a Thai
    codepoint that requires 3 tokens alone means the byte-fallback path failed
    to merge all 3 of its UTF-8 bytes into one token.
    """
    unique = {ch for ch in text if (not ch.isascii()) and ch.isprintable()}
    counts = [(ch, len(encoder(ch))) for ch in unique]
    counts.sort(key=lambda kv: (kv[1], kv[0]), reverse=True)
    return counts[:top_n]


@validated
def verdict_for(lang: str, m: AuditMetrics) -> str:
    """Map (lang, metrics) → verdict string per the M0/§5/R4 ladder.

    Precedence (highest-severity wins):
        Thai:  FAIL_S5 (byte-fallback ≥ 0.20)
            >  FAIL_M0 (tokens/char ≥ 2.0)
            >  WARN_R4 (tokens/char ≥ 2.5 OR byte-fallback ≥ 0.15)
            >  PASS
        FR / DE / EN:  WARN (tokens/char > 1.5)  >  PASS
    """
    if lang == "th":
        if m.single_byte_token_fraction >= THAI_FAIL_S5_SINGLE_BYTE_FRAC:
            return VERDICT_FAIL_S5
        if m.tokens_per_char >= THAI_FAIL_M0_TOKENS_PER_CHAR:
            return VERDICT_FAIL_M0
        if (
            m.tokens_per_char >= THAI_WARN_R4_TOKENS_PER_CHAR
            or m.single_byte_token_fraction >= THAI_WARN_R4_SINGLE_BYTE_FRAC
        ):
            return VERDICT_WARN_R4
        return VERDICT_PASS
    if lang in _LATIN_LANGS:
        if m.tokens_per_char > LATIN_WARN_TOKENS_PER_CHAR:
            return VERDICT_WARN_LATIN
        return VERDICT_PASS
    # Unknown lang — neutral; downstream renderer will display whatever lang
    # tag was given. Don't FAIL on unknown langs so --languages stays extensible.
    return VERDICT_PASS


@validated
def overall_exit_code(verdicts: list[str]) -> int:
    """0 all PASS · 1 any WARN · 3 any FAIL (M0 or §5).

    CI-actionable; the markdown artifact is written either way.
    """
    if any(v in (VERDICT_FAIL_M0, VERDICT_FAIL_S5) for v in verdicts):
        return 3
    if any(v in (VERDICT_WARN_R4, VERDICT_WARN_LATIN) for v in verdicts):
        return 1
    return 0


@validated
def render_markdown(
    rows: list[tuple[LanguageSample, AuditMetrics, str]],
    model_id: str,
    git_commit: str,
    run_id: str | None,
    generated_at: datetime | None = None,
) -> str:
    """Render the M0 artifact markdown for ``eval/tokenizer_audit.md``.

    ``rows`` is the per-language tuple ``(sample, metrics, verdict)``.
    """
    ts = (generated_at or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")
    run_id_line = run_id or "n/a"

    header = [
        "# Tokenizer Audit — Qwen2.5-7B-Instruct (Thai / FR / DE)",
        "",
        f"**Generated**: {ts}  ·  **Model**: `{model_id}`  ·  "
        f"**Git commit**: `{git_commit}`  ·  **MLflow run**: `{run_id_line}`",
        "",
        "## Verdict",
        "",
    ]
    verdict_lines = [f"- **{sample.lang}** — `{verdict}`" for sample, _, verdict in rows]
    methodology = [
        "",
        "## Methodology",
        "",
        "- **Byte-fallback (BBPE-flavoured)**: fraction of emitted tokens whose decoded "
        "UTF-8 form is exactly one byte. Qwen2.5 uses byte-level BPE — there is no "
        "SentencePiece byte-fallback marker — so this operational definition is the "
        "actionable proxy for §5's `>20%` hard stop.",
        "- **Sample**: pypdfium2 plain-text extraction of a page band midway through "
        "each source PDF. Extraction fidelity is irrelevant to per-character "
        "tokenization metrics (Thai codepoints are 3-byte UTF-8 regardless of layout).",
        f"- **Thresholds**: Thai `tokens/char ≥ {THAI_FAIL_M0_TOKENS_PER_CHAR}` → "
        f"`FAIL_M0` · `single_byte_frac ≥ {THAI_FAIL_S5_SINGLE_BYTE_FRAC}` → "
        f"`FAIL_S5` · `tokens/char ≥ {THAI_WARN_R4_TOKENS_PER_CHAR}` OR "
        f"`single_byte_frac ≥ {THAI_WARN_R4_SINGLE_BYTE_FRAC}` → `WARN_R4` · "
        f"FR/DE/EN `tokens/char > {LATIN_WARN_TOKENS_PER_CHAR}` → `WARN`.",
        "",
        "## Per-language metrics",
        "",
        "| Lang | Chars | Tokens | tokens/char | tokens/word | single-byte frac "
        "| p50 | p90 | p99 | Verdict |",
        "|------|------:|-------:|------------:|------------:|-----------------:"
        "|----:|----:|----:|---------|",
    ]
    metric_rows = [
        f"| {sample.lang} | {metrics.total_chars} | {metrics.total_tokens} "
        f"| {metrics.tokens_per_char:.3f} "
        f"| {(f'{metrics.tokens_per_word:.2f}' if metrics.tokens_per_word is not None else 'n/a')} "
        f"| {metrics.single_byte_token_fraction:.3f} "
        f"| {metrics.token_char_len_p50:.1f} | {metrics.token_char_len_p90:.1f} "
        f"| {metrics.token_char_len_p99:.1f} | `{verdict}` |"
        for sample, metrics, verdict in rows
    ]
    thai_rows = [r for r in rows if r[0].lang == "th"]
    thai_section: list[str] = []
    if thai_rows and thai_rows[0][1].top_fragmented_chars:
        thai_section = [
            "",
            "## Top-fragmented characters (Thai diagnostic)",
            "",
            "Tokens required to encode each codepoint **standalone**. A Thai char "
            "that needs 3 tokens alone means BBPE never merged its 3 UTF-8 bytes "
            "into a multi-byte token — i.e. true byte-fallback for that char.",
            "",
            "| Char | Codepoint | Tokens alone |",
            "|------|-----------|-------------:|",
        ]
        thai_section.extend(
            f"| `{ch}` | U+{ord(ch):04X} | {n} |" for ch, n in thai_rows[0][1].top_fragmented_chars
        )
    sources_header = [
        "",
        "## Sources",
        "",
        "| Lang | Framework | Filename | SHA-256 | Page range |",
        "|------|-----------|----------|---------|------------|",
    ]
    source_rows = [
        f"| {sample.lang} | {sample.source_framework} | `{sample.source_filename}` "
        f"| `{sample.source_sha256[:16]}…` | [{sample.page_range[0]}, {sample.page_range[1]}) |"
        for sample, _, _ in rows
    ]
    footer = [
        "",
        "## Reproduce",
        "",
        "```",
        "cd envs/audit && uv run python scripts/run_tokenizer_audit.py",
        "```",
        "",
    ]
    return "\n".join(
        header
        + verdict_lines
        + methodology
        + metric_rows
        + thai_section
        + sources_header
        + source_rows
        + footer
    )


@validated
def render_csv_rows(
    rows: list[tuple[LanguageSample, AuditMetrics, str]],
) -> list[dict[str, str]]:
    """Flatten ``(sample, metrics, verdict)`` triples to CSV-ready dicts.

    Schema is stable; downstream comparison after a base-model swap diffs on it.
    """
    return [
        {
            "lang": sample.lang,
            "framework": sample.source_framework,
            "filename": sample.source_filename,
            "source_sha256": sample.source_sha256,
            "page_start": str(sample.page_range[0]),
            "page_end": str(sample.page_range[1]),
            "total_chars": str(metrics.total_chars),
            "total_tokens": str(metrics.total_tokens),
            "tokens_per_char": f"{metrics.tokens_per_char:.6f}",
            "tokens_per_word": (
                f"{metrics.tokens_per_word:.6f}" if metrics.tokens_per_word is not None else ""
            ),
            "single_byte_token_fraction": f"{metrics.single_byte_token_fraction:.6f}",
            "token_char_len_p50": f"{metrics.token_char_len_p50:.3f}",
            "token_char_len_p90": f"{metrics.token_char_len_p90:.3f}",
            "token_char_len_p99": f"{metrics.token_char_len_p99:.3f}",
            "verdict": verdict,
        }
        for sample, metrics, verdict in rows
    ]


@validated
def write_csv(rows: list[dict[str, str]], buffer: Any) -> None:
    """Write ``render_csv_rows`` output into a text-mode writer with stable column order.

    ``buffer`` is any object with a ``.write(str)`` method — both
    ``io.StringIO`` (tests) and the ``TextIOWrapper`` returned by
    ``Path.open(..., 'w')`` (CLI). Annotated ``Any`` because pydantic's
    ``@validated`` rejects abstract ``typing.TextIO`` at runtime, and the
    ``csv.DictWriter`` contract is duck-typed on ``.write`` anyway.
    """
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
