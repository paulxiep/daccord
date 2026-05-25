# `data/` — pipeline stages

The data pipeline writes through these subdirectories in sequence. Only **committed** dirs are tracked in git; the rest are produced locally and reproducible from the lockfile + scripts.

| Dir | Committed? | Stage | Populated by |
|---|---|---|---|
| `raw/` | no | Source PDFs | tier 1D (PDF download) |
| `ingest/` | no | Parsed markdown | tier 4 (Marker / Thai parser) |
| `registry/` | no | Per-framework valid citation IDs | tier 5 |
| `ensemble/` | no | Multi-model candidate generations (CoT JSON) | tier 7A |
| `tiering/` | no | HIGH/MEDIUM/LOW/SALVAGE classifications | tier 8 |
| `gold/` | **yes** | Hand-validated mapping pairs (gold_v*.jsonl) | tiers 7C, 8, 9 |
| `splits/` | no | Train/val/test JSONL splits with dataset SHA | tier 9 |

The committed-vs-ignored split is enforced by [.gitignore](../.gitignore). The hand-validated gold set + dataset hash is the durable artifact — everything else can be regenerated.
