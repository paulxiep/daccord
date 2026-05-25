# D'accord

Cross-jurisdiction regulatory clause mapping. Privacy MVP across SEA + EU; legal-domain extensible.

D'accord ("agreed" in French) is a small specialized language model fine-tuned to find the parallel provisions of a given regulatory clause across other jurisdictions, with article-level citations. Given a control requirement in (e.g.) GDPR, it returns the analogous requirement in PDPA-SG / PDPA-TH / Loi Informatique et Libertés / BDSG with the exact section ID and a one-sentence justification.

**MVP scope (privacy)**: 8 data-privacy framework families spanning SEA-4 + EU-spine + UK + DE + FR. Operational-resilience deferred to v1; other legal domains (employment, AML/KYC, consumer protection) extensible via the same pipeline.

**Why this exists**: Cross-jurisdiction compliance teams currently re-map controls manually across 4–10 frameworks when a business expands to new jurisdictions. Frontier LLMs hallucinate citations and miss SEA-specific framework references. A small specialized model with citation-faithful output + native validation (Thai + French via the author) closes that gap.

**Interface**: a side-by-side comparison demo — input a privacy clause, see the fine-tuned d'accord output, the retrieval-baseline output, and (when input is in the eval set) the gold answer in parallel columns, each with a provenance tag (`gold-retrieval` / `fine-tune-generalization` / `no-confident-match`) and clickable citations. One-click CSV export of the comparison row, so the output drops straight into a compliance team's control matrix.

## Scope (MVP)

**SEA-4**:
| Jurisdiction | Frameworks | Native validation |
|---|---|---|
| Singapore | PDPA-SG | English |
| Thailand | PDPA-TH + Royal Gazette amendments | **Thai (author reads natively)** |
| Philippines | DPA 2012 | English |
| Malaysia | PDPA-MY | English |

**EU-spine + UK + DE + FR**:
| Jurisdiction | Frameworks | Native validation |
|---|---|---|
| EU-level | GDPR | English |
| United Kingdom | UK-GDPR, DPA 2018, ICO guidance | English |
| Germany | BDSG | German + English translations |
| France | Loi Informatique et Libertés + CNIL guidance | **French (author reads partially)** |

## Architecture

| Stage | Responsibility |
|---|---|
| `data/ingest/` | PDF → markdown via Marker (locked for both EN and TH; Thai bake-off result: marker recall=1.0 precision=1.0 reading-order=5.0 vs typhoon reading-order=4.0 — see [data/parser_bakeoff/summary.md](data/parser_bakeoff/summary.md)) |
| `data/registry/` | Per-framework valid citation IDs extracted from parsed markdown — deterministic key for ensemble filtering |
| `data/ensemble/` | Multi-model candidate generation — four open-weight models via free-tier APIs (Llama 4 Scout via Groq, Qwen 3-32B via Groq/Cerebras, Gemini 3.1 Flash Lite via Google AI Studio, DeepSeek V3); chain-of-thought JSON output; citation IDs constrained to registry |
| `data/tiering/` | HIGH/MEDIUM/LOW/SALVAGE classification (deterministic via registry agreement) |
| `data/gold/` | Hand-validated mapping pairs (target 500–1000) — committed |
| `training/` | QLoRA training (Qwen3-8B base) + MLflow autolog |
| `eval/` | Three-tier scoring (citation exact match + LLM-as-judge semantic + ~100-example human spot-check) across four comparators including a retrieval baseline; stratified by in-domain vs out-of-domain ([eval/README.md](eval/README.md)) |
| `publish/` | S3 packaging in SageMaker-compatible layout — bundles QLoRA adapter + retrieval index + embedder snapshot + custom inference handler |
| `src/daccord/serving/` | `HybridRouter` (retrieval-first, QLoRA fallback, per-response provenance tagging) shared between local demo and SageMaker handler |
| `consumer/` | Side-by-side comparison + CSV export (Streamlit), backed by hybrid retrieval+fine-tune SageMaker endpoint with per-response provenance tagging |

## Methodology

**Dataset construction (~50% of effort)**:
1. **Seed from public crosswalks** — NIST 800-53 ↔ ISO 27001 (published by NIST, free) as authoritative anchors.
2. **Citation registry extraction** — per-framework structured TOC of valid section/article IDs from parsed markdown. Avoids the "LLM says `GDPR Art. 32` but Marker emitted `### 32. Security of processing`" mismatch under naive substring matching.
3. **Multi-model ensemble with constrained citations + chain-of-thought** — each model emits `{source_mechanism, target_mechanism, mapping_justification, citation_id}` with citation_id restricted to the target framework's registry. Four diverse open-weight models served via free-tier APIs: Llama 4 Scout (Groq), Qwen 3-32B (Groq / Cerebras), Gemini 3.1 Flash Lite (Google AI Studio), DeepSeek V3.
4. **Tier classification**:
   - **HIGH** (4/4 agree on citation_id): train set without per-pair labeling; sample 10% for quality audit
   - **MEDIUM** (3/4 agree OR target section consensus with sub-clause disagreement): hand-validate 100%
   - **LOW** (≤2/4 agree): hand-validate 100% → eval set seed (hardest cases)
   - **SALVAGE**: chain-of-thought reads correct but citation_id wrong → manual correct → promote to train
5. **Stratified human spot-check on HIGH-tier per jurisdiction** — catches systematic ensemble bias (real risk for under-represented SEA regs where all four models may share the same blind spots).

**Training**: QLoRA on Qwen3-8B, MLflow-tracked, local on RTX 5080 (16GB VRAM is sufficient for 8B QLoRA at 4-bit NF4).

**Eval** — three-tier scoring across four comparator models:
- **Tier 1 — Citation exact match**: deterministic, cheap; top-1 and top-3.
- **Tier 2 — LLM-as-judge semantic match**: Llama 4 Scout via Groq free tier scores substance match, mitigating exact-match penalty for valid paraphrasings.
- **Tier 3 — Human spot-check** (~100 examples): quantifies judge accuracy; calibrates Tier 2 scores.

**Comparators**: fine-tuned d'accord vs base Qwen 3-8B vs Llama 4 Scout (Groq) vs Qwen 3-32B (Groq) vs Gemini 3.1 Flash Lite (Google AI Studio) vs **retrieval baseline** (sentence-transformers MPNet + FAISS over train-split source clauses). The retrieval baseline answers the architectural question "could you have just done retrieval?" with data. The Qwen-3-32B comparator additionally asks "would the newer-and-bigger same-family model already beat us without fine-tune?".

**Stratification**: each eval pass runs twice — once on **in-domain** pairs (val/test inputs whose source clauses are cosine-near a train-split clause; retrieval-friendly) and once on **out-of-domain** pairs (jurisdiction-disjoint held-out requirement areas; where the fine-tune's generalization should pay off). Slice tag goes to MLflow run metadata, not the per-row CSV (CSV row contract stays stable per [eval/README.md](eval/README.md)).

Per-jurisdiction + per-language breakdowns are aggregated from CSV rows at read time to quantify the SEA/FR/DE differentiation.

## Current State

Phase 1 (local validation) in progress. Milestone tags follow the [development plan](docs/development_plan.md).

- **[M0 partial — 2026-05-25]** Eval bar locked. 20-pair toy gold built (9/20 author-verified, 11/20 claude-extract-only pending paraphrase pass — deferred to pre-tier-4 MR). Tokenizer audit PASS on th/fr/de/en for **Qwen3-8B** (the chosen QLoRA base, swapped from Qwen2.5-7B mid-session). Thai parser bake-off shipped — Marker locked. Baseline numbers on partial-verified gold include Qwen 3-8B (local, NF4), Llama 4 Scout (Groq), Qwen 3-32B (Groq), and Gemini 3.1 Flash Lite; judged by Llama 4 Scout. See [eval/baseline_toy.csv](eval/baseline_toy.csv) for the row-by-row numbers.
- **[2C — 2026-05-24]** Tokenizer audit (`th` 0.575 tok/char · `fr` 0.520 · `de` 0.303 · `en` 0.213 — all PASS). R4 resolved.
- **[2D — 2026-05-24]** Thai parser bake-off (5-page sample, 61 hand-verified Thai citations) — Marker recall=1.0 precision=1.0 reading-order=5.0 vs Typhoon-OCR reading-order=4.0; Marker locked for both EN and TH. R1 resolved.
- **[2B — 2026-05-23]** Eval harness end-to-end (citation match + LLM-as-judge, 14-col CSV contract, MLflow nested-run logging).
- **[1A–1D — 2026-05-22]** Repo skeleton + MLflow plumbing + per-provider RPD caps (Groq + Gemini + Cerebras + DeepSeek) + 13-PDF corpus across 8 jurisdictions on disk.
- **[2026-05-25 refactor]** Dev environment migrated to Docker Compose (Linux containers); consumer pivoted from chatbot → side-by-side comparison + CSV export (`src/daccord/serving/HybridRouter`); shared 10-RPM throttle + Gemini transient-error retry + Groq APIError safety net wired into all API clients; eval runner refactored to pair-major iteration (all generators see each pair before moving on) so per-provider RPM density stays low.

Pending the next MR (pre-tier-4): 11 claude-extract-only rows pending paraphrase semantic pass in `data/gold/toy_v1.jsonl`; FR Loi I+L native-validation coverage at 2/3 pairs (decide accept-vs-repoint); tier 4 (full corpus parse to markdown via Marker).

## Development environment

**All development runs in Linux containers via Docker Compose.** Two thin Dockerfiles (CPU + CUDA) back six compose services (`root`, `eval`, `audit`, `bakeoff`, `baseline`, `consumer`). Host requirements: Docker Desktop (WSL2 backend) + an NVIDIA Windows driver for the GPU services — no CUDA Toolkit install needed on the host (bundled in the CUDA image).

Quick start:

```bash
docker compose build root eval                       # CPU image (first time only)
docker compose run --rm root uv sync                 # shared daccord lib
docker compose run --rm eval uv sync                 # tier 2B eval harness
docker compose run --rm eval uv run pytest           # 78/78 should pass
```

Per-env Python split: root/eval/audit/baseline/consumer on 3.14; bakeoff on 3.13 (held back by marker-pdf's `pillow<11` ceiling). Each service's `working_dir` is set to its env folder in `docker-compose.yml`, so `pytest`/`ruff`/`pyright` pick up the right `pyproject.toml` without a `cd`.

## Tech Stack

- **Base model**: Qwen3-8B (multilingual: Thai, French, German, English — tokenizer audit PASS on all four; see [eval/tokenizer_audit.md](eval/tokenizer_audit.md))
- **Training**: QLoRA via PEFT
- **MLOps tracking**: MLflow
- **PDF processing**: Marker (locked for both EN and TH after a 5-page Thai bake-off vs Typhoon-OCR — both hit perfect citation extraction; Marker preferred for ~2× faster wall time and noise-free body output free of Royal Gazette page-header chrome)
- **Ensemble labelers**: four open-weight models via free-tier APIs — Llama 4 Scout (Groq), Qwen 3-32B (Groq / Cerebras), Gemini 3.1 Flash Lite (Google AI Studio), DeepSeek V3
- **Eval judge**: Llama 4 Scout via Groq free tier (bumped from Llama 3.3-70B on 2026-05-25 for stronger judging signal; self-judging-bias note in the M0 baseline CSV when `groq` is in the generator pool)
- **Retrieval baseline**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` + `faiss-cpu` (FAISS index over train-split source clauses; also reused at serving time by the hybrid router)
- **Hybrid serving**: `HybridRouter` (retrieval-first, QLoRA fallback, per-response provenance tagging) — shared between the local Streamlit demo and the SageMaker custom inference handler
- **Demo consumer**: Streamlit side-by-side comparison UI with CSV export
- **Deployment**: boto3-direct to SageMaker real-time endpoint (`ml.g5.xlarge`)

## Roadmap

- [Development plan (Phase 1 → Phase 2)](docs/development_plan.md) — tiers, milestones, gates, risk register

| Version | Date | Focus |
|---|---|---|
| **v0 MVP** | 2026 Q2 | Privacy: SEA-4 + EU(spine+UK+DE+FR); QLoRA fine-tune on Qwen3-8B; three-tier eval (citation exact match + LLM-as-judge + human spot-check) with retrieval baseline + in/out-of-domain stratification; hybrid serving (retrieval + fine-tune fallback with provenance tagging); Streamlit side-by-side comparison + CSV export |
| **v1** | TBD | Operational-resilience extension: MAS TRM, BOT IT, OJK POJK, BNM RMiT, BSP, DORA, EBA, PRA SS1/21, BaFin BAIT/MaRisk |
| **v2+** | TBD | Additional legal domains (employment, AML/KYC, consumer protection) and additional jurisdictions |

## Known Limitations

- **Browser-print PDFs for UK + FR**: UK-GDPR, UK DPA 2018, and FR Loi Informatique et Libertés have no scraper-friendly consolidated PDFs (legislation.gov.uk and Légifrance expose only HTML). The corpus falls back to browser print-to-PDF for these three sources — 5–60× larger files with embedded page chrome. Risk R8 in the development plan; spot-check planned at tier 4 against a regulator-issued reference (BDSG).
- **Gemini free-tier daily cap**: `gemini-3.1-flash-lite` ships 15 RPM / 500 RPD on the free tier. The earlier `gemini-2.5-flash` daily cap was as low as 20 RPD on some accounts, which exhausted mid-baseline; the project standardised on 3.1 Flash Lite and the Llama 4 Scout judge sidesteps any per-minute spikes via the 10-RPM global throttle + transient-error retry layer + pair-major iteration (per-provider density stays at ~1 call per pair-cycle, well under any single provider's cap).
- **No `envs/training/` env yet**: tier 10A (QLoRA training script) will add a 7th compose service. RTX 5080 (16 GB VRAM) is sufficient for 7B QLoRA but a small-sweep at M3 will validate OOM headroom; Unsloth fallback documented in the dev plan if VRAM is tight at full `max_seq_len`.

---

### Keywords

- **Language**: `Python`
- **Domain**: `Cross-Jurisdiction Regulatory Mapping` · `Privacy Compliance` · `RegTech` · `Legal NLP` · `Citation-Faithful Output`
- **Frameworks (MVP)**: `GDPR` · `UK-GDPR` · `DPA 2018` · `BDSG` · `Loi Informatique et Libertés` · `CNIL guidance` · `PDPA-SG` · `PDPA-TH` · `DPA 2012 (PH)` · `PDPA-MY`
- **ML & MLOps**: `QLoRA Fine-Tuning` · `PEFT` · `MLflow` · `LoRA Adapter` · `Multi-Model Ensemble Labeling` · `Weak Supervision` · `Chain-of-Thought Structured Output` · `Citation Registry Constraint` · `Three-Tier Evaluation` · `LLM-as-Judge` · `Human Spot-Check Calibration`
- **Base / Ensemble Models**: `Qwen3-8B (base)` · `Qwen3-32B` · `Llama 4 Scout (17B × 16E MoE)` · `Gemini 3.1 Flash Lite` · `DeepSeek V3` (all open-weight; free-tier-served)
- **PDF / Layout**: `Marker (ViT layout, locked for EN + TH)` · `LlamaParse (fallback)` · `Citation Registry Extraction`
- **Deployment**: `AWS SageMaker (endpoint hosting via boto3, custom inference handler w/ HybridRouter)` · `Streamlit side-by-side comparison + CSV export UI`
