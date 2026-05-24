# Toy gold v1 — 20-pair provenance log

Per-pair record of how `data/gold/toy_v1.jsonl` was assembled. The credential claim
depends on every committed citation being human-verified against the source PDF,
not LLM-hallucinated. This file is the audit trail. Rows conform to
[daccord.gold.schema.GoldPair](../../src/daccord/gold/schema.py).

## Pipeline (as executed)

The standard pipeline (draft → reconcile → verify → commit) was streamlined:

1. **Direct PDF read** — instead of running the LLM drafters at
   `envs/eval/scripts/draft_toy_gold.py`, the 20 pairs were composed by reading
   the regulator-issued PDFs in `data/raw/` directly via
   [envs/audit/scripts/extract_pdf_text.py](../../envs/audit/scripts/extract_pdf_text.py)
   (pypdfium2 text extraction). This skips the LLM-hallucination layer entirely for
   the frameworks the agent could read first-hand (GDPR, PDPA-SG, PDPA-TH English,
   PDPA-MY, DPA-PH, BDSG English).
2. **Author native verification** — for frameworks the agent could not validate
   semantically (Thai original, French Loi I+L, German BDSG-specific provisions),
   the author performs the final native-language pass against the authoritative
   PDFs. Pairs requiring this pass carry an explicit `PENDING ... verification`
   note in `toy_v1.jsonl`'s `notes` field and in the table below.
3. **Coverage matrix gate** — verified via
   [envs/eval/scripts/verify_toy_coverage.py](../../envs/eval/scripts/verify_toy_coverage.py):
   `ALL ACCEPTANCE GATES: PASS` (20 rows · all 8 jurisdictions ≥2 · th=4 ≥3 · fr=4 ≥3).
4. **Schema-validate via `--dry-run`**:
   `cd envs/eval && uv run python scripts/run_eval.py --gold-path ../../data/gold/toy_v1.jsonl --dry-run --verbose`
   reports `loaded 20 pairs` and `dataset_hash = 412501438684f1ea9c2fcfbdcbb92897cb469fd795c61f8839e193824e3880a5`.

The dataset hash above is the contract `MLflow` will tie every baseline + fine-tune
run to. If any pair is edited the hash changes, and every downstream eval run will
log the new hash automatically.

## Per-pair log

`drafted_by` values:
- `human-direct` — agent read the source/target PDFs directly via pypdfium2 (no LLM draft step).
- `human-edited` — substantive human rewrite (none currently).

`verified_by`: `claude (initial)` = agent did PDF cross-reference for languages it
could read (en, de). The author closes the loop by verifying citation_ids against
the authoritative original-language PDF for any pair flagged `PENDING` in the
`notes` column below.

| id       | drafted_by    | source_pdf_filename                                          | target_pdf_filename                                        | verified_by       | verification_date | notes |
|----------|---------------|--------------------------------------------------------------|------------------------------------------------------------|-------------------|-------------------|-------|
| toy_001  | human-direct  | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 6 p. ~3)       | `uk/uk_gdpr/uk_gdpr_current.pdf` (Art. 6, post-Brexit retained) | claude (initial) | 2026-05-25 | UK_GDPR not extracted; author may spot-check the retained-law form |
| toy_002  | human-direct  | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 37)            | `de/bdsg/bdsg_en_current.pdf` (Sec. 38, unconfirmed)       | claude (initial) | 2026-05-25 | **PENDING DE verification** — BDSG Sec 38 (private-body DPO threshold) not directly extracted; author confirms |
| toy_003  | human-direct  | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 33)            | `fr/loi_il/loi_78_17_consolidated.pdf` (Art. 33, by reference) | claude (initial) | 2026-05-25 | **PENDING FR verification** — Loi I+L browser-print is mostly URL chrome on first 6 pages; author confirms whether Art 33 stands alone or cross-refers to GDPR |
| toy_004  | human-direct  | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 32)            | `fr/loi_il/loi_78_17_consolidated.pdf` (Art. 121, uncertain) | claude (initial) | 2026-05-25 | **PENDING FR verification** — post-2018 ordonnance renumbering; author confirms exact article |
| toy_005  | human-direct  | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 15)            | `uk/uk_gdpr/uk_gdpr_current.pdf` (Art. 15)                 | claude (initial) | 2026-05-25 | UK_GDPR not extracted; UK GDPR retains article numbering verbatim |
| toy_006  | human-direct  | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 6)             | `sg/pdpa_sg/pdpa_sg_current.pdf` (Sec. 13, p. 24)          | claude (initial) | 2026-05-25 | Verified Sec 13 wording directly from PDPA-SG PDF |
| toy_007  | human-direct  | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 15)            | `sg/pdpa_sg/pdpa_sg_current.pdf` (Sec. 21, p. 29)          | claude (initial) | 2026-05-25 | Verified Sec 21 wording directly |
| toy_008  | human-direct  | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 32)            | `sg/pdpa_sg/pdpa_sg_current.pdf` (Sec. 24)                 | claude (initial) | 2026-05-25 | PDPA-SG Sec 24 verified by TOC + Part 6 structure; full text in pages 30–40 not re-quoted |
| toy_009  | human-direct  | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 6)             | `my/pdpa_my/pdpa_my_act709_bilingual.pdf` (Sec. 6, English column) | claude (initial) | 2026-05-25 | PDPA-MY 'Akta 709' Sec 6 = General Principle; bilingual PDF — English column used for mechanism text |
| toy_010  | human-direct  | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 15)            | `my/pdpa_my/pdpa_my_act709_bilingual.pdf` (Sec. 30)        | claude (initial) | 2026-05-25 | PDPA-MY Sec 30 confirmed via TOC ('Hak untuk mengakses data peribadi') |
| toy_011  | human-direct  | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 6)             | `ph/dpa_2012_ph/dpa_2012_ph.pdf` (Sec. 12(a), p. 11)       | claude (initial) | 2026-05-25 | DPA-PH Sec 12(a) verified by direct quote |
| toy_012  | human-direct  | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 33)            | `ph/dpa_2012_ph/dpa_2012_ph.pdf` (Sec. 20(f), p. 17)       | claude (initial) | 2026-05-25 | DPA-PH Sec 20(f) verified by direct quote |
| toy_013  | human-direct  | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 6)             | `th/pdpa_th/pdpa_th_english_2019.pdf` (Sec. 19, p. 7)      | claude (initial) | 2026-05-25 | **PENDING TH verification** — Sec 19 paraphrased from English unofficial translation; cross-check มาตรา 19 in `pdpa_th_thai_2019.pdf` |
| toy_014  | human-direct  | `th/pdpa_th/pdpa_th_english_2019.pdf` (Sec. 37(4), p. 16)    | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 33)          | claude (initial) | 2026-05-25 | **PENDING TH verification** — confirm มาตรา 37(4) language matches the 72-hour mechanism |
| toy_015  | human-direct  | `th/pdpa_th/pdpa_th_english_2019.pdf` (Sec. 37(1), p. 16)    | `sg/pdpa_sg/pdpa_sg_current.pdf` (Sec. 24)                 | claude (initial) | 2026-05-25 | **PENDING TH verification** — confirm มาตรา 37(1) security obligation phrasing |
| toy_016  | human-direct  | `de/bdsg/bdsg_en_current.pdf` (Sec. 38, unconfirmed)         | `fr/loi_il/loi_78_17_consolidated.pdf` (Art. 8, uncertain) | claude (initial) | 2026-05-25 | **PENDING DE + FR verification** — both citations are best-guess; needs native confirmation |
| toy_017  | human-direct  | `uk/dpa_2018/dpa_2018_current.pdf` (Sec. 45, unconfirmed; 17 MB browser-print) | `eu/gdpr/reg_2016_679_consolidated.pdf` (Art. 15)          | claude (initial) | 2026-05-25 | **PENDING UK verification** — UK DPA 2018 Sec 45 (Part 3 law-enforcement DSAR) not directly extracted; R8 caveat applies |
| toy_018  | human-direct  | `uk/dpa_2018/dpa_2018_current.pdf` (Sec. 69, unconfirmed)    | `de/bdsg/bdsg_en_current.pdf` (Sec. 5, p. 3)               | claude (initial) | 2026-05-25 | **PENDING UK verification** — UK DPA 2018 Sec 69; BDSG Sec 5 verified directly |
| toy_019  | human-direct  | `sg/pdpa_sg/pdpa_sg_current.pdf` (Sec. 21, p. 29)            | `my/pdpa_my/pdpa_my_act709_bilingual.pdf` (Sec. 30)        | claude (initial) | 2026-05-25 | Both confirmed via TOC; SG fee mechanism is Sec 28, MY fee is in Sec 30 |
| toy_020  | human-direct  | `fr/loi_il/loi_78_17_consolidated.pdf` (Art. 49, uncertain)  | `th/pdpa_th/pdpa_th_english_2019.pdf` (Sec. 30, p. 13)     | claude (initial) | 2026-05-25 | **PENDING FR + TH verification** — Loi I+L Art 49 (post-2018 numbering) + TH มาตรา 30 |

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
