"""Unit tests for daccord.tokenizer_audit — the 2C tokenizer audit library.

Pure-math tests over synthetic tokens. No HF download, no PDF I/O, no MLflow.
End-to-end verification of the real Qwen2.5 tokenizer is left to running
``envs/audit/scripts/run_tokenizer_audit.py`` and inspecting ``eval/tokenizer_audit.md``.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from daccord.tokenizer_audit import (
    LATIN_WARN_TOKENS_PER_CHAR,
    THAI_FAIL_M0_TOKENS_PER_CHAR,
    THAI_FAIL_S5_SINGLE_BYTE_FRAC,
    THAI_WARN_R4_SINGLE_BYTE_FRAC,
    THAI_WARN_R4_TOKENS_PER_CHAR,
    VERDICT_FAIL_M0,
    VERDICT_FAIL_S5,
    VERDICT_PASS,
    VERDICT_WARN_LATIN,
    VERDICT_WARN_R4,
    AuditMetrics,
    LanguageSample,
    compute_audit_metrics,
    overall_exit_code,
    render_csv_rows,
    render_markdown,
    verdict_for,
    write_csv,
)

# ---------- helpers --------------------------------------------------------


def make_metrics(
    *,
    total_chars: int = 100,
    total_tokens: int = 100,
    tokens_per_char: float = 1.0,
    tokens_per_word: float | None = 1.0,
    single_byte_token_fraction: float = 0.0,
    p50: float = 1.0,
    p90: float = 2.0,
    p99: float = 3.0,
    top_fragmented_chars: list[tuple[str, int]] | None = None,
) -> AuditMetrics:
    return AuditMetrics(
        total_chars=total_chars,
        total_tokens=total_tokens,
        tokens_per_char=tokens_per_char,
        tokens_per_word=tokens_per_word,
        single_byte_token_fraction=single_byte_token_fraction,
        token_char_len_p50=p50,
        token_char_len_p90=p90,
        token_char_len_p99=p99,
        top_fragmented_chars=top_fragmented_chars or [],
    )


def make_sample(lang: str = "th") -> LanguageSample:
    return LanguageSample(
        lang=lang,
        source_framework="pdpa_th",
        source_filename="pdpa_th_thai_2019.pdf",
        source_sha256="9fd1027c2e5ad93029a01d68d52d524750c1e8946b97f12fb230c11959da2f70",
        page_range=(5, 13),
        char_count=4096,
    )


def _id_to_piece(tid: int) -> str:
    """Stub decoder mapping id → 1-byte ASCII char, capping at 0x7f.

    Lets us deterministically construct token streams whose single-byte
    fraction is exactly known to the test.
    """
    return chr(tid % 0x80)


def _id_to_two_byte_piece(tid: int) -> str:
    """Decoder stub: every id decodes to a 2-char piece (Latin-extended)."""
    base = (tid % 100) + 0x100
    return chr(base) + chr(base + 1)


def _trivial_encoder(s: str) -> list[int]:
    """Each codepoint → 1 token. Used when fragmentation diagnostic is uninteresting."""
    return [ord(ch) for ch in s]


def _three_byte_encoder(s: str) -> list[int]:
    """Each non-ASCII codepoint → 3 tokens (simulates pure byte-fallback)."""
    out: list[int] = []
    for ch in s:
        if ch.isascii():
            out.append(ord(ch))
        else:
            out.extend([ord(ch), ord(ch) + 1, ord(ch) + 2])
    return out


# ---------- compute_audit_metrics ------------------------------------------


def test_compute_metrics_all_single_byte() -> None:
    text = "hello"  # 5 chars
    token_ids = [ord(c) for c in text]  # 5 tokens
    m = compute_audit_metrics(text, token_ids, _id_to_piece, _trivial_encoder)
    assert m.total_chars == 5
    assert m.total_tokens == 5
    assert m.tokens_per_char == pytest.approx(1.0)
    assert m.single_byte_token_fraction == pytest.approx(1.0)


def test_compute_metrics_mixed_single_byte_and_multi() -> None:
    # 6 tokens: 3 single-byte + 3 two-byte
    text = "abcdef"
    token_ids = [1, 2, 3, 100, 101, 102]

    # Custom decoder: ids 1-3 → 1-byte; ids 100+ → 2-byte
    def decoder(tid: int) -> str:
        return "x" if tid < 50 else "ää"

    m = compute_audit_metrics(text, token_ids, decoder, _trivial_encoder)
    assert m.total_tokens == 6
    assert m.single_byte_token_fraction == pytest.approx(0.5)


def test_compute_metrics_empty_text_no_zero_division() -> None:
    m = compute_audit_metrics("", [], _id_to_piece, _trivial_encoder)
    assert m.tokens_per_char == 0.0
    assert m.tokens_per_word == 0.0
    assert m.single_byte_token_fraction == 0.0
    assert m.top_fragmented_chars == []


def test_compute_metrics_no_word_split_for_thai() -> None:
    text = "กขคงจฉชซ"  # 8 Thai chars, no spaces
    token_ids = list(range(16))  # arbitrary
    m = compute_audit_metrics(
        text, token_ids, _id_to_piece, _trivial_encoder, word_split_supported=False
    )
    assert m.tokens_per_word is None


def test_compute_metrics_top_fragmented_picks_worst_codepoints() -> None:
    # All 3 chars are non-ASCII; encoder gives them 1, 2, 3 tokens respectively.
    text = "αβγ"

    def encoder(s: str) -> list[int]:
        if s == "α":
            return [1]
        if s == "β":
            return [1, 2]
        if s == "γ":
            return [1, 2, 3]
        return [0]

    m = compute_audit_metrics(text, [0], _id_to_piece, encoder, top_fragmented_n=2)
    chars = [c for c, _ in m.top_fragmented_chars]
    counts = [n for _, n in m.top_fragmented_chars]
    assert chars[0] == "γ"
    assert counts == [3, 2]


def test_compute_metrics_top_fragmented_ignores_ascii() -> None:
    text = "abc αβγ"
    m = compute_audit_metrics(text, [0], _id_to_piece, _three_byte_encoder, top_fragmented_n=10)
    chars = {c for c, _ in m.top_fragmented_chars}
    assert chars.isdisjoint(set("abc "))
    assert chars == {"α", "β", "γ"}


# ---------- verdict_for ----------------------------------------------------


def test_verdict_thai_fail_s5_when_byte_fallback_exceeds_threshold() -> None:
    m = make_metrics(tokens_per_char=1.0, single_byte_token_fraction=0.21)
    assert verdict_for("th", m) == VERDICT_FAIL_S5


def test_verdict_thai_fail_m0_when_tokens_per_char_exceeds_threshold() -> None:
    m = make_metrics(tokens_per_char=2.01, single_byte_token_fraction=0.05)
    assert verdict_for("th", m) == VERDICT_FAIL_M0


def test_verdict_thai_fail_s5_wins_when_both_fail() -> None:
    # Precedence: FAIL_S5 reported when byte-fallback also crosses threshold.
    m = make_metrics(tokens_per_char=2.5, single_byte_token_fraction=0.25)
    assert verdict_for("th", m) == VERDICT_FAIL_S5


def test_verdict_thai_warn_r4_high_tokens_per_char() -> None:
    # >= 2.5 tokens/char but <2.0... wait, threshold is 2.0 FAIL. So WARN_R4
    # triggers only on byte-fallback ≥ 0.15 when tokens/char <2.0, OR on
    # tokens/char ≥ 2.5 (which also triggers FAIL_M0 first → so WARN_R4 is
    # effectively the byte-fallback-only warning path).
    m = make_metrics(tokens_per_char=1.5, single_byte_token_fraction=0.16)
    assert verdict_for("th", m) == VERDICT_WARN_R4


def test_verdict_thai_pass_when_below_all_thresholds() -> None:
    m = make_metrics(tokens_per_char=1.2, single_byte_token_fraction=0.05)
    assert verdict_for("th", m) == VERDICT_PASS


def test_verdict_latin_pass_when_under_threshold() -> None:
    m = make_metrics(tokens_per_char=0.3)
    for lang in ("fr", "de", "en"):
        assert verdict_for(lang, m) == VERDICT_PASS


def test_verdict_latin_warn_when_over_threshold() -> None:
    m = make_metrics(tokens_per_char=LATIN_WARN_TOKENS_PER_CHAR + 0.01)
    assert verdict_for("fr", m) == VERDICT_WARN_LATIN


def test_verdict_unknown_lang_is_neutral_pass() -> None:
    m = make_metrics(tokens_per_char=99.0)
    assert verdict_for("xx", m) == VERDICT_PASS


# ---------- overall_exit_code ----------------------------------------------


def test_exit_code_all_pass() -> None:
    assert overall_exit_code([VERDICT_PASS, VERDICT_PASS]) == 0


def test_exit_code_any_warn_is_one() -> None:
    assert overall_exit_code([VERDICT_PASS, VERDICT_WARN_LATIN]) == 1
    assert overall_exit_code([VERDICT_WARN_R4]) == 1


def test_exit_code_any_fail_is_three() -> None:
    assert overall_exit_code([VERDICT_WARN_LATIN, VERDICT_FAIL_M0]) == 3
    assert overall_exit_code([VERDICT_FAIL_S5]) == 3


# ---------- render_markdown ------------------------------------------------


def test_render_markdown_contains_all_languages() -> None:
    rows = [
        (
            make_sample("th"),
            make_metrics(
                tokens_per_char=1.2,
                single_byte_token_fraction=0.05,
                tokens_per_word=None,
                top_fragmented_chars=[("ก", 3), ("ข", 2)],
            ),
            VERDICT_PASS,
        ),
        (
            LanguageSample(
                lang="fr",
                source_framework="loi_il",
                source_filename="loi_78_17_consolidated.pdf",
                source_sha256="8803c88720db82fa78ea387104e5199bb9daa564eeb4002d8f53d63212c82970",
                page_range=(20, 28),
                char_count=8192,
            ),
            make_metrics(tokens_per_char=0.28),
            VERDICT_PASS,
        ),
    ]
    md = render_markdown(
        rows,
        model_id="Qwen/Qwen2.5-7B-Instruct",
        git_commit="abc123def456",
        run_id="run-001",
        generated_at=datetime(2026, 5, 23, 10, 0, tzinfo=UTC),
    )
    assert "Qwen/Qwen2.5-7B-Instruct" in md
    assert "abc123def456" in md
    assert "run-001" in md
    assert "## Verdict" in md
    assert "**th**" in md
    assert "**fr**" in md
    assert "Top-fragmented characters" in md
    assert "ก" in md
    assert "loi_78_17_consolidated.pdf" in md
    assert str(THAI_FAIL_M0_TOKENS_PER_CHAR) in md
    assert str(THAI_FAIL_S5_SINGLE_BYTE_FRAC) in md
    assert str(THAI_WARN_R4_TOKENS_PER_CHAR) in md
    assert str(THAI_WARN_R4_SINGLE_BYTE_FRAC) in md
    assert "## Reproduce" in md


def test_render_markdown_omits_thai_section_when_no_thai_row() -> None:
    rows = [
        (
            LanguageSample(
                lang="de",
                source_framework="bdsg",
                source_filename="bdsg_de_current.pdf",
                source_sha256="c02a85a398b18b953f90d1852612806ddfdf097af327f86a289de4c9ce27d25f",
                page_range=(5, 13),
                char_count=4096,
            ),
            make_metrics(tokens_per_char=0.3),
            VERDICT_PASS,
        ),
    ]
    md = render_markdown(rows, model_id="Qwen/Qwen2.5-7B-Instruct", git_commit="x", run_id=None)
    assert "Top-fragmented characters" not in md


# ---------- render_csv_rows / write_csv ------------------------------------


def test_render_csv_rows_schema_stable_across_rows() -> None:
    rows = [
        (make_sample("th"), make_metrics(tokens_per_word=None), VERDICT_PASS),
        (make_sample("fr"), make_metrics(tokens_per_word=1.6), VERDICT_PASS),
    ]
    csv_rows = render_csv_rows(rows)
    assert len(csv_rows) == 2
    assert csv_rows[0].keys() == csv_rows[1].keys()
    assert "tokens_per_char" in csv_rows[0]
    assert "single_byte_token_fraction" in csv_rows[0]
    assert "verdict" in csv_rows[0]
    assert csv_rows[0]["tokens_per_word"] == ""  # None → empty cell
    assert csv_rows[1]["tokens_per_word"] != ""


def test_write_csv_emits_header_and_rows() -> None:
    rows = [(make_sample("th"), make_metrics(), VERDICT_PASS)]
    buf = io.StringIO()
    write_csv(render_csv_rows(rows), buf)
    text = buf.getvalue()
    assert text.startswith("lang,framework,filename,")
    assert "th,pdpa_th,pdpa_th_thai_2019.pdf" in text
    assert "PASS" in text


def test_write_csv_empty_rows_writes_nothing() -> None:
    buf = io.StringIO()
    write_csv([], buf)
    assert buf.getvalue() == ""


# ---------- @validated runtime contract ------------------------------------


def test_validated_rejects_non_str_lang() -> None:
    with pytest.raises(ValidationError):
        verdict_for(123, make_metrics())  # type: ignore[arg-type]


def test_validated_rejects_wrong_metrics_shape() -> None:
    with pytest.raises(ValidationError):
        verdict_for("th", "not-a-metrics")  # type: ignore[arg-type]
