# Toy gold v1 — 20-pair coverage matrix

> ⚠ **`toy_v1.jsonl` is an UNVALIDATED DRAFT** (status 2026-05-25): 0 / 20 pairs
> have been human-verified against authoritative PDFs. This file documents the
> *coverage plan*; **citation correctness** is the separate concern flagged in
> [toy_v1_provenance.md](toy_v1_provenance.md) — read its top banner before
> consuming `toy_v1.jsonl` as M0 gold.

Pre-plan for [toy_v1.jsonl](toy_v1.jsonl) per the 2A creation plan in
[docs/development_plan.md §9.2](../../docs/development_plan.md). Committed so the
20-pair selection is defensible: every M0 baseline + M4 per-jurisdiction breakdown
metric reads against this matrix.

## Axes

- **Jurisdiction coverage** — every jurisdiction in `{eu, uk, de, fr, sg, th, ph, my}`
  appears as source or target in ≥2 pairs so per-jurisdiction MLflow metrics aren't
  single-point.
- **Native-language validation moat** — PDPA-TH ≥3 pairs (author validates Thai
  natively); FR Loi I+L ≥3 pairs (author validates French). These two are the
  credential's headline claim; under-sampling them hollows out M4.
- **Concept axes** — `consent`, `dsar` (data subject access rights), `breach`
  (notification), `security`, `dpo` (DPO appointment / controller duties). Each
  appears ≥3 times.
- **Difficulty mix** — 5 easy (EU-spine clones) / 8 medium (GDPR ↔ SEA-4 / UK / EU
  bilateral) / 7 hard (non-EU-spine pairs, e.g. PDPA-TH ↔ PDPA-SG, BDSG ↔ Loi I+L).

## Matrix

| id       | difficulty | concept   | source        | target        | source_jur | target_jur |
|----------|------------|-----------|---------------|---------------|-----------|-----------|
| toy_001  | easy       | consent   | gdpr          | uk_gdpr       | eu        | uk        |
| toy_002  | easy       | dpo       | gdpr          | bdsg          | eu        | de        |
| toy_003  | easy       | breach    | gdpr          | loi_il        | eu        | fr        |
| toy_004  | easy       | security  | gdpr          | loi_il        | eu        | fr        |
| toy_005  | easy       | dsar      | gdpr          | uk_gdpr       | eu        | uk        |
| toy_006  | medium     | consent   | gdpr          | pdpa_sg       | eu        | sg        |
| toy_007  | medium     | dsar      | gdpr          | pdpa_sg       | eu        | sg        |
| toy_008  | medium     | security  | gdpr          | pdpa_sg       | eu        | sg        |
| toy_009  | medium     | consent   | gdpr          | pdpa_my       | eu        | my        |
| toy_010  | medium     | dsar      | gdpr          | pdpa_my       | eu        | my        |
| toy_011  | medium     | consent   | gdpr          | dpa_2012_ph   | eu        | ph        |
| toy_012  | medium     | breach    | gdpr          | dpa_2012_ph   | eu        | ph        |
| toy_013  | medium     | consent   | gdpr          | pdpa_th       | eu        | th        |
| toy_014  | hard       | breach    | pdpa_th       | gdpr          | th        | eu        |
| toy_015  | hard       | security  | pdpa_th       | pdpa_sg       | th        | sg        |
| toy_016  | hard       | dpo       | bdsg          | loi_il        | de        | fr        |
| toy_017  | hard       | dsar      | dpa_2018      | gdpr          | uk        | eu        |
| toy_018  | hard       | dpo       | dpa_2018      | bdsg          | uk        | de        |
| toy_019  | hard       | dsar      | pdpa_sg       | pdpa_my       | sg        | my        |
| toy_020  | hard       | dsar      | loi_il        | pdpa_th       | fr        | th        |

## Coverage tallies

**Jurisdictions** (source + target counts; gate ≥2 each):

| jur | count | pairs |
|---|---:|---|
| eu  | 14 | 001–013 (as source or target on most), 014, 017 |
| uk  | 4  | 001, 005, 017, 018 |
| de  | 3  | 002, 016, 018 |
| fr  | 4  | 003, 004, 016, 020 |
| sg  | 5  | 006, 007, 008, 015, 019 |
| th  | 4  | 013, 014, 015, 020 |
| ph  | 2  | 011, 012 |
| my  | 3  | 009, 010, 019 |

All jurisdictions ≥2 ✓ ; **th=4 ≥3 ✓** ; **fr=4 ≥3 ✓** ; native-language moat satisfied.

**Concept axes** (gate ≥3 each):

| concept   | count | pairs |
|---|---:|---|
| consent   | 5 | 001, 006, 009, 011, 013 |
| dsar      | 6 | 005, 007, 010, 017, 019, 020 |
| breach    | 3 | 003, 012, 014 |
| security  | 3 | 004, 008, 015 |
| dpo       | 3 | 002, 016, 018 |

All axes ≥3 ✓.

**Difficulty mix**: 5 easy / 8 medium / 7 hard (matches the drafter's default mix).

## Source-PDF assignment

Per the language-source rules in [toy_v1_provenance.md](toy_v1_provenance.md), each
non-English framework has an authoritative original-language PDF and an English
cross-reference. For citation_id verification I cross-check both; for
`*_mechanism` text I cite from the indicated PDF below (using English where the
schema's per-row `*_language` field can record it accurately).

| framework    | authoritative PDF                                        | language used for `mechanism` text |
|--------------|----------------------------------------------------------|-----------------------------------:|
| gdpr         | `data/raw/eu/gdpr/reg_2016_679_consolidated.pdf`         | en |
| uk_gdpr      | `data/raw/uk/uk_gdpr/uk_gdpr_current.pdf`                | en |
| dpa_2018     | `data/raw/uk/dpa_2018/dpa_2018_current.pdf`              | en |
| bdsg         | `data/raw/de/bdsg/bdsg_de_current.pdf` (authoritative)   | en (from `bdsg_en_current.pdf`) — author cross-checks DE |
| loi_il       | `data/raw/fr/loi_il/loi_78_17_consolidated.pdf` (only)   | fr — author validates natively |
| pdpa_sg      | `data/raw/sg/pdpa_sg/pdpa_sg_current.pdf`                | en |
| pdpa_th      | `data/raw/th/pdpa_th/pdpa_th_thai_2019.pdf` (authoritative) | en (from `pdpa_th_english_2019.pdf`) — author cross-checks TH |
| dpa_2012_ph  | `data/raw/ph/dpa_2012_ph/dpa_2012_ph.pdf`                | en |
| pdpa_my      | `data/raw/my/pdpa_my/pdpa_my_act709_bilingual.pdf`       | en (English column of bilingual) |
