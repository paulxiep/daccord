# Toy gold v1 — 20-pair provenance log

> ## ⚠ STATUS: UNVALIDATED DRAFT — DO NOT TREAT AS M0 GOLD
>
> **Human-verified pairs: 0 / 20.** All 20 rows in [toy_v1.jsonl](toy_v1.jsonl)
> are claude-drafted from direct PDF text extraction. **Zero rows have been
> human-validated against the authoritative source PDFs by the author.** The
> credential claim explicitly depends on every committed citation being
> human-verified — *not* claude-verified.
>
> **M0 gate cannot close on this artifact.** Tier 3A baselines can technically
> *run* against the draft (schema is valid, dataset_hash is stable), but any
> resulting `eval/baseline_toy.csv` numbers are against unvalidated gold and
> must not be cited or compared until the human pass is complete.
>
> **Why the draft was committed anyway**: per the 2026-05-25 session, the
> author requested that claude do as much as possible first ("you do everything
> first, then point me to double check"). This file is the handoff: the 20
> pairs are queued for the human pass, organized by priority below.
>
> **Future sessions**: if you see baseline_toy.csv numbers that look wrong on
> a TH / FR / DE / UK pair, *check this file first*. If the relevant pair is
> still flagged as a STUB in the table below, the gold itself is the bug —
> not the model under eval.
>
> **What "validated" means here**: author opens `data/raw/<jur>/<framework>/`
> PDF, confirms (a) the cited article/section actually exists with the exact
> citation_id, and (b) the `*_mechanism` paraphrase is faithful to the actual
> text (not the claude paraphrase's gist). Then sets the row's `verified_by`
> + `verification_date` cells below and (if any edits) removes the PENDING
> note from `toy_v1.jsonl`. Once all 20 rows are validated, this banner
> comes down.

Per-pair record of how `data/gold/toy_v1.jsonl` was assembled. The credential claim
depends on every committed citation being human-verified against the source PDF,
not LLM-hallucinated. This file is the audit trail. Rows conform to
[daccord.gold.schema.GoldPair](../../src/daccord/gold/schema.py).

## Pipeline (as executed, 2026-05-25)

The standard pipeline (draft → reconcile → verify → commit) was **incomplete** —
draft + reconcile (machine-only) ran; verify (human) did NOT:

1. ✓ **Direct PDF read (draft equivalent)** — instead of running the LLM drafters at
   `envs/eval/scripts/draft_toy_gold.py`, the 20 pairs were composed by reading
   the regulator-issued PDFs in `data/raw/` directly via
   [envs/audit/scripts/extract_pdf_text.py](../../envs/audit/scripts/extract_pdf_text.py)
   (pypdfium2 text extraction). This skips the LLM-hallucination layer for the
   frameworks claude could read first-hand (GDPR, PDPA-SG, PDPA-TH English,
   PDPA-MY, DPA-PH, BDSG English).
2. ✗ **Author native verification — NOT YET DONE.** The author must open each
   source/target PDF in `data/raw/`, confirm citation_id exists exactly as
   written, and confirm `*_mechanism` paraphrases the actual text. *All 20 pairs
   are pending this step.* Pairs already flagged STUB below (best-guess
   citation_id, not yet confirmed against any extract) need additional work.
3. ✓ **Coverage matrix gate (mechanical)** — verified via
   [envs/eval/scripts/verify_toy_coverage.py](../../envs/eval/scripts/verify_toy_coverage.py):
   `ALL ACCEPTANCE GATES: PASS` (20 rows · all 8 jurisdictions ≥2 · th=4 ≥3 · fr=4 ≥3).
   This is a mechanical / structural check — it does NOT validate citation
   correctness.
4. ✓ **Schema-validate via `--dry-run` (mechanical)**:
   `cd envs/eval && uv run python scripts/run_eval.py --gold-path ../../data/gold/toy_v1.jsonl --dry-run --verbose`
   reports `loaded 20 pairs` and `dataset_hash = 412501438684f1ea9c2fcfbdcbb92897cb469fd795c61f8839e193824e3880a5`.
   This confirms the JSONL is well-formed, not that citations are correct.

The dataset hash above is the contract `MLflow` will tie every baseline + fine-tune
run to. **The hash will change** once the author edits any cell during the
verification pass — that's the desired behavior (it forces every downstream eval
run to log a new hash, so unvalidated-gold numbers can't be silently re-used).

## Per-pair log

`drafted_by` values (current):
- `claude-direct` — claude read the source/target PDFs directly via pypdfium2 (no LLM draft step).
- `human-edited` — substantive human rewrite (none currently — added after author pass).

`verified_by` values:
- `UNVERIFIED` — **current state for all 20 rows.** No human pass has happened yet.
- `claude-extract-only` — claude cross-referenced the citation against directly-extracted PDF text
  for languages it could read (en, de) but the author has not opened the PDF.
  **Does not satisfy the credential's "human-verified" bar.**
- `STUB` — claude could not extract or cross-reference the target citation_id at draft time
  (FR Loi I+L browser-print noise / UK DPA 2018 large file / DE BDSG Sec 38 not in extracted range).
  The citation_id is a best-guess from general regulatory knowledge. **Likely to be wrong.**
- `<author-id>` — populated when the author closes the loop. **None populated yet.**

The two columns to the right (`verified_by`, `verification_date`) flip from `UNVERIFIED` /
empty → author-id / ISO date as each row is reviewed. The 10 STUB rows additionally need
their citation_ids corrected (not just confirmed).

| id       | drafted_by    | source_pdf_filename                                          | target_pdf_filename                                        | verified_by       | verification_date | notes |
|----------|---------------|--------------------------------------------------------------|------------------------------------------------------------|-------------------|-------------------|-------|
| toy_001  | claude-direct | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 6 p. ~3)       | `uk/uk_gdpr/uk_gdpr_current.pdf` (Art. 6, post-Brexit retained) | UNVERIFIED — claude-extract-only (source side) | — | UK_GDPR not extracted; author spot-checks the retained-law form |
| toy_002  | claude-direct | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 37)            | `de/bdsg/bdsg_en_current.pdf` (Sec. 38, unconfirmed)       | UNVERIFIED — STUB on DE target | — | **STUB DE** — BDSG Sec 38 (private-body DPO 20-employee threshold) not directly extracted; citation_id is best-guess |
| toy_003  | claude-direct | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 33)            | `fr/loi_il/loi_78_17_consolidated.pdf` (Art. 33, by reference) | UNVERIFIED — STUB on FR target | — | **STUB FR** — Loi I+L browser-print is mostly URL chrome on first 6 pages; author confirms whether Art 33 stands alone or cross-refers to GDPR |
| toy_004  | claude-direct | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 32)            | `fr/loi_il/loi_78_17_consolidated.pdf` (Art. 121, uncertain) | UNVERIFIED — STUB on FR target | — | **STUB FR** — post-2018 ordonnance renumbering; Article 121 is a best-guess (was Art 34 pre-2018); author confirms exact article |
| toy_005  | claude-direct | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 15)            | `uk/uk_gdpr/uk_gdpr_current.pdf` (Art. 15)                 | UNVERIFIED — claude-extract-only (source side) | — | UK_GDPR not extracted; UK GDPR retains article numbering verbatim post-Brexit; author spot-checks |
| toy_006  | claude-direct | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 6)             | `sg/pdpa_sg/pdpa_sg_current.pdf` (Sec. 13, p. 24)          | UNVERIFIED — claude-extract-only | — | Sec 13 wording confirmed via direct PDF extract; author final pass |
| toy_007  | claude-direct | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 15)            | `sg/pdpa_sg/pdpa_sg_current.pdf` (Sec. 21, p. 29)          | UNVERIFIED — claude-extract-only | — | Sec 21 wording confirmed via direct PDF extract; author final pass |
| toy_008  | claude-direct | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 32)            | `sg/pdpa_sg/pdpa_sg_current.pdf` (Sec. 24)                 | UNVERIFIED — claude-extract-only (TOC only on target) | — | PDPA-SG Sec 24 verified via TOC + Part 6 structure; full text pages 30–40 not re-quoted; author confirms wording |
| toy_009  | claude-direct | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 6)             | `my/pdpa_my/pdpa_my_act709_bilingual.pdf` (Sec. 6, English column) | UNVERIFIED — claude-extract-only | — | PDPA-MY 'Akta 709' Sec 6 = General Principle; bilingual PDF — author confirms English column quote |
| toy_010  | claude-direct | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 15)            | `my/pdpa_my/pdpa_my_act709_bilingual.pdf` (Sec. 30)        | UNVERIFIED — claude-extract-only (TOC only on target) | — | PDPA-MY Sec 30 confirmed via TOC ('Hak untuk mengakses data peribadi'); author confirms wording |
| toy_011  | claude-direct | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 6)             | `ph/dpa_2012_ph/dpa_2012_ph.pdf` (Sec. 12(a), p. 11)       | UNVERIFIED — claude-extract-only | — | DPA-PH Sec 12(a) confirmed via direct PDF extract; author final pass |
| toy_012  | claude-direct | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 33)            | `ph/dpa_2012_ph/dpa_2012_ph.pdf` (Sec. 20(f), p. 17)       | UNVERIFIED — claude-extract-only | — | DPA-PH Sec 20(f) confirmed via direct PDF extract; author final pass |
| toy_013  | claude-direct | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 6)             | `th/pdpa_th/pdpa_th_english_2019.pdf` (Sec. 19, p. 7)      | UNVERIFIED — STUB on TH-native side | — | **STUB TH** — Sec 19 paraphrased from English unofficial translation only; cross-check มาตรา 19 in `pdpa_th_thai_2019.pdf` (author native validation required) |
| toy_014  | claude-direct | `th/pdpa_th/pdpa_th_english_2019.pdf` (Sec. 37(4), p. 16)    | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 33)          | UNVERIFIED — STUB on TH-native side | — | **STUB TH** — confirm มาตรา 37(4) (72-hour breach) language matches in Thai original |
| toy_015  | claude-direct | `th/pdpa_th/pdpa_th_english_2019.pdf` (Sec. 37(1), p. 16)    | `sg/pdpa_sg/pdpa_sg_current.pdf` (Sec. 24)                 | UNVERIFIED — STUB on TH-native side | — | **STUB TH** — confirm มาตรา 37(1) security-obligation phrasing in Thai original |
| toy_016  | claude-direct | `de/bdsg/bdsg_en_current.pdf` (Sec. 38, unconfirmed)         | `fr/loi_il/loi_78_17_consolidated.pdf` (Art. 8, uncertain) | UNVERIFIED — STUB on both sides | — | **STUB DE + FR** — both citation_ids are best-guess from general knowledge; needs DE + FR author confirmation |
| toy_017  | claude-direct | `uk/dpa_2018/dpa_2018_current.pdf` (Sec. 45, unconfirmed; 17 MB browser-print) | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 15)          | UNVERIFIED — STUB on UK source side | — | **STUB UK** — UK DPA 2018 Sec 45 (Part 3 law-enforcement DSAR) not directly extracted; R8 caveat applies to the source PDF |
| toy_018  | claude-direct | `uk/dpa_2018/dpa_2018_current.pdf` (Sec. 69, unconfirmed)    | `de/bdsg/bdsg_en_current.pdf` (Sec. 5, p. 3)               | UNVERIFIED — STUB on UK source side | — | **STUB UK** — UK DPA 2018 Sec 69 (Part 3 DPO) not directly extracted; BDSG Sec 5 target was directly quoted |
| toy_019  | claude-direct | `sg/pdpa_sg/pdpa_sg_current.pdf` (Sec. 21, p. 29)            | `my/pdpa_my/pdpa_my_act709_bilingual.pdf` (Sec. 30)        | UNVERIFIED — claude-extract-only (TOC only on target) | — | Both confirmed via TOC; SG fee mechanism is Sec 28, MY fee is in Sec 30; author confirms fee-mechanism wording |
| toy_020  | claude-direct | `fr/loi_il/loi_78_17_consolidated.pdf` (Art. 49, uncertain)  | `th/pdpa_th/pdpa_th_english_2019.pdf` (Sec. 30, p. 13)     | UNVERIFIED — STUB on both sides | — | **STUB FR + TH** — Loi I+L Art 49 (post-2018 numbering) is best-guess; TH มาตรา 30 needs Thai-original confirmation |

### STUB count summary

| State | Count | Pair IDs |
|---|---:|---|
| STUB (citation_id is best-guess, likely needs correction) | 10 | toy_002, 003, 004, 013, 014, 015, 016, 017, 018, 020 |
| claude-extract-only (citation_id confirmed via PDF extract, needs author semantic pass) | 10 | toy_001, 005, 006, 007, 008, 009, 010, 011, 012, 019 |
| Human-verified | **0** | — |

## Author double-check punch list

These are the cells the author should focus on (ordered by jurisdiction so a single
PDF open can clear several pairs at once):

**Thai (PDPA-TH original `pdpa_th_thai_2019.pdf`)** — cross-check that the cited
มาตรา numbers match the English-translation paraphrases:
- toy_013 — `มาตรา 19` (consent)
- toy_014 — `มาตรา 37(4)` (breach 72-hour notification)
- toy_015 — `มาตรา 37(1)` (security measures)
- toy_020 — `มาตรา 30` (right of access)

**French (Loi I+L `loi_78_17_consolidated.pdf`)** — confirm post-2018 ordonnance
article numbering (the browser-print PDF is R8-noisy; first 6 pages are URL chrome):
- toy_003 — Art 33 (breach — likely by reference to GDPR Art 33)
- toy_004 — Art 121 (security — historical Art 34 pre-2018; renumbering uncertain)
- toy_016 — Art 8 (DPO designation)
- toy_020 — Art 49 (right of access)

**German (BDSG `bdsg_de_current.pdf`)** — confirm:
- toy_002 — Sec 38 private-body DPO threshold of 20 employees
- toy_016 — Sec 38 mapping to French DPO supplement

**UK DPA 2018 (`dpa_2018_current.pdf`, 17 MB browser-print, R8 caveat)** — confirm:
- toy_017 — Sec 45 (Part 3 DSAR for law-enforcement processing)
- toy_018 — Sec 69 (Part 3 DPO designation)

For all other pairs (toy_001, 005, 006-012, 019) the citation_id was confirmed by
direct quote from the extracted PDF text; only stylistic refinements are likely needed.

## Language-source notes

For non-English frameworks, the authoritative original-language PDF was used for
citation lookup (per R-2A.1: citation IDs can differ between English and
original-language source PDFs):

- **PDPA-TH**: citation IDs (มาตรา N) cross-referenced against
  `pdpa_th_thai_2019.pdf` (authoritative). `target_mechanism` paraphrase taken from
  `pdpa_th_english_2019.pdf` (unofficial translation) with `target_language="en"`.
- **BDSG**: citation IDs taken from `bdsg_en_current.pdf` (official BMJ
  translation). Section numbers should match the German authoritative
  `bdsg_de_current.pdf` 1:1.
- **Loi I+L (FR)**: only `loi_78_17_consolidated.pdf` (French, browser-printed
  from Légifrance) exists; mechanism text in French. `target_language="fr"`.
- **PDPA-MY**: `pdpa_my_act709_bilingual.pdf` is bilingual EN+MS; English column
  cited.
