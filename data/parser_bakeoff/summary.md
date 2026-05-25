# Thai parser bake-off — summary

Source PDF: `data\raw\th\pdpa_th\pdpa_th_thai_2019.pdf`
Sample pages (0-indexed): [1, 11, 16, 39, 42]
Parsers compared: ['marker', 'typhoon']

## Per-parser aggregates

| Parser | Pages | Recall | Precision | Reading order | Structure preserved |
|---|---:|---:|---:|---:|---:|
| marker | 5 | 1.000 | 1.000 | 5.000 | 1.000 |
| typhoon | 5 | 1.000 | 1.000 | 4.000 | 1.000 |

## Picked-winner rule

1. Highest reading-order mean wins (Thai-reader judgment, weight 1).
2. Tie-break on highest citation recall (M1 registry extraction depends on this).
3. If both parsers' reading-order mean is below 3.0, invoke the M1 cut criterion
   (drop Royal Gazette amendments, retain PDPA-TH core only) per
   `docs/development_plan.md §M1`.

## Decision

**Locked parser: `marker`** — wins on reading-order mean (ro=5.000, recall=1.000) vs. `typhoon` (ro=4.000, recall=1.000).

This is the M1 parser-choice artifact: tier 4 (full-corpus parse to markdown) uses `marker` for Thai PDFs; English-language frameworks already use `marker` by default.
