# Tokenizer Audit — `Qwen/Qwen3-8B` (Thai / FR / DE / EN)

**Generated**: 2026-05-25 15:11 UTC  ·  **Model**: `Qwen/Qwen3-8B`  ·  **Git commit**: `28913b82b78b`  ·  **MLflow run**: `621fc92fb40d41bd8d7fdbe8d3c2ecab`

## Verdict

- **th** — `PASS`
- **fr** — `PASS`
- **de** — `PASS`
- **en** — `PASS`

## Methodology

- **Byte-fallback (BBPE-flavoured)**: fraction of emitted tokens whose decoded UTF-8 form is exactly one byte. The Qwen series uses byte-level BPE — there is no SentencePiece byte-fallback marker — so this operational definition is the actionable proxy for §5's `>20%` hard stop.
- **Sample**: pypdfium2 plain-text extraction of a page band midway through each source PDF. Extraction fidelity is irrelevant to per-character tokenization metrics (Thai codepoints are 3-byte UTF-8 regardless of layout).
- **Thresholds**: Thai `tokens/char ≥ 2.0` → `FAIL_M0` · `single_byte_frac ≥ 0.2` → `FAIL_S5` · `tokens/char ≥ 2.5` OR `single_byte_frac ≥ 0.15` → `WARN_R4` · FR/DE/EN `tokens/char > 1.5` → `WARN`.

## Per-language metrics

| Lang | Chars | Tokens | tokens/char | tokens/word | single-byte frac | p50 | p90 | p99 | Verdict |
|------|------:|-------:|------------:|------------:|-----------------:|----:|----:|----:|---------|
| th | 16735 | 9625 | 0.575 | n/a | 0.055 | 2.0 | 3.0 | 6.0 | `PASS` |
| fr | 1351 | 703 | 0.520 | 4.39 | 0.613 | 1.0 | 4.0 | 9.0 | `PASS` |
| de | 33928 | 10266 | 0.303 | 2.33 | 0.172 | 3.0 | 6.0 | 8.0 | `PASS` |
| en | 20451 | 4364 | 0.213 | 1.34 | 0.180 | 4.0 | 10.0 | 13.0 | `PASS` |

## Top-fragmented characters (Thai diagnostic)

Tokens required to encode each codepoint **standalone**. A Thai char that needs 3 tokens alone means BBPE never merged its 3 UTF-8 bytes into a multi-byte token — i.e. true byte-fallback for that char.

| Char | Codepoint | Tokens alone |
|------|-----------|-------------:|
| `๙` | U+0E59 | 2 |
| `๘` | U+0E58 | 2 |
| `๗` | U+0E57 | 2 |
| `๖` | U+0E56 | 2 |
| `๕` | U+0E55 | 2 |
| `๔` | U+0E54 | 2 |
| `๓` | U+0E53 | 2 |
| `๒` | U+0E52 | 2 |
| `๑` | U+0E51 | 2 |
| `๐` | U+0E50 | 2 |
| `์` | U+0E4C | 1 |
| `้` | U+0E49 | 1 |
| `่` | U+0E48 | 1 |
| `็` | U+0E47 | 1 |
| `ๆ` | U+0E46 | 1 |
| `ไ` | U+0E44 | 1 |
| `ใ` | U+0E43 | 1 |
| `โ` | U+0E42 | 1 |
| `แ` | U+0E41 | 1 |
| `เ` | U+0E40 | 1 |

## Sources

| Lang | Framework | Filename | SHA-256 | Page range |
|------|-----------|----------|---------|------------|
| th | pdpa_th | `pdpa_th_thai_2019.pdf` | `9fd1027c2e5ad930…` | [14, 22) |
| fr | loi_il | `loi_78_17_consolidated.pdf` | `8803c88720db82fa…` | [48, 56) |
| de | bdsg | `bdsg_de_current.pdf` | `c02a85a398b18b95…` | [15, 23) |
| en | gdpr | `reg_2016_679_consolidated.pdf` | `18e6c90c50514302…` | [26, 34) |

## Reproduce

```
cd envs/audit && uv run python scripts/run_tokenizer_audit.py
```
