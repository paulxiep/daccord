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
| `data/ingest/` | PDF → markdown via Marker (locked for both EN and TH after the tier-2D bake-off; see `data/parser_bakeoff/summary.md`) |
| `data/registry/` | Per-framework valid citation IDs extracted from parsed markdown — deterministic key for ensemble filtering |
| `data/ensemble/` | Multi-model candidate generation — four open-source models via free-tier APIs (Llama 3.x 70B via Groq, Qwen 2.5/3 70B via Cerebras, Gemini 2.x Flash via Google AI Studio, DeepSeek V3); chain-of-thought JSON output; citation IDs constrained to registry |
| `data/tiering/` | HIGH/MEDIUM/LOW/SALVAGE classification (deterministic via registry agreement) |
| `data/gold/` | Hand-validated mapping pairs (target 500–1000) — committed |
| `training/` | QLoRA training (Qwen2.5-7B-Instruct base) + MLflow autolog |
| `eval/` | Three-tier scoring (citation exact match + LLM-as-judge semantic + ~100-example human spot-check) across four comparators including a retrieval baseline; stratified by in-domain vs out-of-domain ([eval/README.md](eval/README.md)) |
| `publish/` | S3 packaging in SageMaker-compatible layout — bundles QLoRA adapter + retrieval index + embedder snapshot + custom inference handler |
| `src/daccord/serving/` | `HybridRouter` (retrieval-first, QLoRA fallback, per-response provenance tagging) shared between local demo and SageMaker handler |
| `consumer/` | Side-by-side comparison + CSV export (Streamlit), backed by hybrid retrieval+fine-tune SageMaker endpoint with per-response provenance tagging |

## Methodology

**Dataset construction (~50% of effort)**:
1. **Seed from public crosswalks** — NIST 800-53 ↔ ISO 27001 (published by NIST, free) as authoritative anchors.
2. **Citation registry extraction** — per-framework structured TOC of valid section/article IDs from parsed markdown. Avoids the "LLM says `GDPR Art. 32` but Marker emitted `### 32. Security of processing`" mismatch under naive substring matching.
3. **Multi-model ensemble with constrained citations + chain-of-thought** — each model emits `{source_mechanism, target_mechanism, mapping_justification, citation_id}` with citation_id restricted to the target framework's registry. Four diverse open-source models served via free-tier APIs: Llama 3.x 70B (Groq), Qwen 2.5/3 70B (Cerebras), Gemini 2.x Flash (Google AI Studio), DeepSeek V3.
4. **Tier classification**:
   - **HIGH** (4/4 agree on citation_id): train set without per-pair labeling; sample 10% for quality audit
   - **MEDIUM** (3/4 agree OR target section consensus with sub-clause disagreement): hand-validate 100%
   - **LOW** (≤2/4 agree): hand-validate 100% → eval set seed (hardest cases)
   - **SALVAGE**: chain-of-thought reads correct but citation_id wrong → manual correct → promote to train
5. **Stratified human spot-check on HIGH-tier per jurisdiction** — catches systematic ensemble bias (real risk for under-represented SEA regs where all four models may share the same blind spots).

**Training**: QLoRA on Qwen2.5-7B-Instruct, MLflow-tracked, local on RTX 5080 (16GB VRAM sufficient for 7B QLoRA).

**Eval** — three-tier scoring across four comparator models:
- **Tier 1 — Citation exact match**: deterministic, cheap; top-1 and top-3.
- **Tier 2 — LLM-as-judge semantic match**: Llama 3.x 70B via Groq free tier scores substance match, mitigating exact-match penalty for valid paraphrasings.
- **Tier 3 — Human spot-check** (~100 examples): quantifies judge accuracy; calibrates Tier 2 scores.

**Comparators**: fine-tuned d'accord vs base Qwen 2.5-7B vs Llama 3.x 70B (Groq) vs Gemini 2.x Flash (Google AI Studio) vs **retrieval baseline** (sentence-transformers MPNet + FAISS over train-split source clauses). The retrieval baseline answers the architectural question "could you have just done retrieval?" with data.

**Stratification**: each eval pass runs twice — once on **in-domain** pairs (val/test inputs whose source clauses are cosine-near a train-split clause; retrieval-friendly) and once on **out-of-domain** pairs (jurisdiction-disjoint held-out requirement areas; where the fine-tune's generalization should pay off). Slice tag goes to MLflow run metadata, not the per-row CSV (CSV row contract stays stable per [eval/README.md](eval/README.md)).

Per-jurisdiction + per-language breakdowns are aggregated from CSV rows at read time to quantify the SEA/FR/DE differentiation.

## Status

Planning / early-stage. MVP pipeline: scrape → parse → registry → ensemble generation → tiering → validation → training → eval → deploy.

## Tech Stack

- **Base model**: Qwen2.5-7B-Instruct (multilingual: Thai, French, German, English)
- **Training**: QLoRA via PEFT
- **MLOps tracking**: MLflow
- **PDF processing**: Marker (locked for both EN and TH after a 5-page Thai bake-off vs Typhoon-OCR — both hit perfect citation extraction; Marker preferred for ~2× faster wall time and noise-free body output free of Royal Gazette page-header chrome)
- **Ensemble labelers**: four open-source models via free-tier APIs — Llama 3.x 70B (Groq), Qwen 2.5/3 70B (Cerebras), Gemini 2.x Flash (Google AI Studio), DeepSeek V3
- **Eval judge**: Llama 3.x 70B via Groq free tier
- **Retrieval baseline**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` + `faiss-cpu` (FAISS index over train-split source clauses; also reused at serving time by the hybrid router)
- **Hybrid serving**: `HybridRouter` (retrieval-first, QLoRA fallback, per-response provenance tagging) — shared between the local Streamlit demo and the SageMaker custom inference handler
- **Demo consumer**: Streamlit side-by-side comparison UI with CSV export
- **Deployment**: boto3-direct to SageMaker real-time endpoint (`ml.g5.xlarge`)

## Roadmap

| Version | Focus |
|---|---|
| **v0 MVP** | Privacy: SEA-4 + EU(spine+UK+DE+FR), QLoRA fine-tune, three-tier eval (with retrieval baseline + in/out-of-domain stratification), hybrid serving (retrieval + fine-tune), side-by-side comparison + CSV export demo |
| **v1** | Operational-resilience extension: MAS TRM, BOT IT, OJK POJK, BNM RMiT, BSP, DORA, EBA, PRA SS1/21, BaFin BAIT/MaRisk |
| **v2+** | Additional legal domains (employment, AML/KYC, consumer protection) and additional jurisdictions |

---

### Keywords

- **Language**: `Python`
- **Domain**: `Cross-Jurisdiction Regulatory Mapping` · `Privacy Compliance` · `RegTech` · `Legal NLP` · `Citation-Faithful Output`
- **Frameworks (MVP)**: `GDPR` · `UK-GDPR` · `DPA 2018` · `BDSG` · `Loi Informatique et Libertés` · `CNIL guidance` · `PDPA-SG` · `PDPA-TH` · `DPA 2012 (PH)` · `PDPA-MY`
- **ML & MLOps**: `QLoRA Fine-Tuning` · `PEFT` · `MLflow` · `LoRA Adapter` · `Multi-Model Ensemble Labeling` · `Weak Supervision` · `Chain-of-Thought Structured Output` · `Citation Registry Constraint` · `Three-Tier Evaluation` · `LLM-as-Judge` · `Human Spot-Check Calibration`
- **Base / Ensemble Models**: `Qwen2.5-7B-Instruct` · `Qwen 2.5/3 72B` · `Llama 3.x 70B` · `Gemini 2.x Flash` · `DeepSeek V3` (all open-source / open-weight; free-tier-served)
- **PDF / Layout**: `Marker (ViT layout, locked for EN + TH)` · `LlamaParse (fallback)` · `Citation Registry Extraction`
- **Deployment**: `AWS SageMaker (endpoint hosting via boto3)` · `Streamlit (chatbot UI)`
