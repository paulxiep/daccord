# D'accord

Cross-jurisdiction regulatory clause mapping. Privacy MVP across SEA + EU; legal-domain extensible.

D'accord ("agreed" in French) is a small specialized language model fine-tuned to find the parallel provisions of a given regulatory clause across other jurisdictions, with article-level citations. Given a control requirement in (e.g.) GDPR, it returns the analogous requirement in PDPA-SG / PDPA-TH / Loi Informatique et Libertés / BDSG with the exact section ID and a one-sentence justification.

**MVP scope (privacy)**: 8 data-privacy framework families spanning SEA-4 + EU-spine + UK + DE + FR. Operational-resilience deferred to v1; other legal domains (employment, AML/KYC, consumer protection) extensible via the same pipeline.

**Why this exists**: Cross-jurisdiction compliance teams currently re-map controls manually across 4–10 frameworks when a business expands to new jurisdictions. Frontier LLMs hallucinate citations and miss SEA-specific framework references. A small specialized model with citation-faithful output + native validation (Thai + French via the author) closes that gap.

**Interface**: a conversational chatbot — *"Ask D'accord about a privacy regulation; it answers with the parallel provisions in other jurisdictions, with citations."*

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
| `data/ingest/` | PDF → markdown via Marker (default) / PaddleOCR+coords / Typhoon (Thai bake-off candidate) |
| `data/registry/` | Per-framework valid citation IDs extracted from parsed markdown — deterministic key for ensemble filtering |
| `data/ensemble/` | Multi-model candidate generation (Claude 3.5 Sonnet + GPT-4o + Qwen2.5-72B); chain-of-thought JSON output; citation IDs constrained to registry |
| `data/tiering/` | HIGH/MEDIUM/LOW/SALVAGE classification (deterministic via registry agreement) |
| `data/gold/` | Hand-validated mapping pairs (target 500–1000) — committed |
| `training/` | QLoRA training (Qwen2.5-7B-Instruct base) + MLflow autolog |
| `eval/` | Three-tier: citation exact match + LLM-as-judge semantic + ~100-example human spot-check |
| `publish/` | S3 packaging in SageMaker-compatible layout |
| `consumer/` | "Ask D'accord" chatbot interface — conversational Q&A with citations |

## Methodology

**Dataset construction (~50% of effort)**:
1. **Seed from public crosswalks** — NIST 800-53 ↔ ISO 27001 (published by NIST, free) as authoritative anchors.
2. **Citation registry extraction** — per-framework structured TOC of valid section/article IDs from parsed markdown. Avoids the "LLM says `GDPR Art. 32` but Marker emitted `### 32. Security of processing`" mismatch under naive substring matching.
3. **Multi-model ensemble with constrained citations + chain-of-thought** — each model emits `{source_mechanism, target_mechanism, mapping_justification, citation_id}` with citation_id restricted to the target framework's registry. Three diverse models: Claude 3.5 Sonnet (Bedrock), GPT-4o, Qwen2.5-72B-Instruct (Together/Modal).
4. **Tier classification**:
   - **HIGH** (3/3 agree on citation_id): train set without per-pair labeling; sample 10% for quality audit
   - **MEDIUM** (2/3 agree OR target section consensus with sub-clause disagreement): hand-validate 100%
   - **LOW** (<2 agree): hand-validate 100% → eval set seed (hardest cases)
   - **SALVAGE**: chain-of-thought reads correct but citation_id wrong → manual correct → promote to train
5. **Stratified human spot-check on HIGH-tier per jurisdiction** — catches systematic ensemble bias (real risk for under-represented SEA regs where all three models may share the same blind spots).

**Training**: QLoRA on Qwen2.5-7B-Instruct, MLflow-tracked, local on RTX 5080 (16GB VRAM sufficient for 7B QLoRA).

**Eval** — three-tier:
- **Tier 1 — Citation exact match**: deterministic, cheap; top-1 and top-3.
- **Tier 2 — LLM-as-judge semantic match**: Claude 3.5 Haiku or GPT-4o-mini scores substance match, mitigating exact-match penalty for valid paraphrasings.
- **Tier 3 — Human spot-check** (~100 examples): quantifies judge accuracy; calibrates Tier 2 scores.

Per-jurisdiction + per-language breakdowns to quantify the SEA/FR/DE differentiation.

## Status

Planning / early-stage. MVP pipeline: scrape → parse → registry → ensemble generation → tiering → validation → training → eval → deploy.

## Tech Stack

- **Base model**: Qwen2.5-7B-Instruct (multilingual: Thai, French, German, English)
- **Training**: QLoRA via PEFT
- **MLOps tracking**: MLflow
- **PDF processing**: Marker (default), PaddleOCR + spatial-clustering (fallback for non-Latin scripts), Typhoon (Thai-native bake-off candidate)
- **Ensemble labelers**: Claude 3.5 Sonnet (Bedrock), GPT-4o (OpenAI), Qwen2.5-72B-Instruct (Together / Modal)
- **Eval judge**: Claude 3.5 Haiku or GPT-4o-mini
- **Demo consumer**: Streamlit chatbot UI
- **Deployment**: [Caravan](https://github.com/paulxiep/caravan) `aws-sagemaker-llm` target (SageMaker real-time endpoint)

## Roadmap

| Version | Focus |
|---|---|
| **v0 MVP** | Privacy: SEA-4 + EU(spine+UK+DE+FR), QLoRA fine-tune, three-tier eval, chatbot demo consumer |
| **v1** | Operational-resilience extension: MAS TRM, BOT IT, OJK POJK, BNM RMiT, BSP, DORA, EBA, PRA SS1/21, BaFin BAIT/MaRisk |
| **v2+** | Additional legal domains (employment, AML/KYC, consumer protection) and additional jurisdictions |

---

### Keywords

- **Language**: `Python`
- **Domain**: `Cross-Jurisdiction Regulatory Mapping` · `Privacy Compliance` · `RegTech` · `Legal NLP` · `Citation-Faithful Output`
- **Frameworks (MVP)**: `GDPR` · `UK-GDPR` · `DPA 2018` · `BDSG` · `Loi Informatique et Libertés` · `CNIL guidance` · `PDPA-SG` · `PDPA-TH` · `DPA 2012 (PH)` · `PDPA-MY`
- **ML & MLOps**: `QLoRA Fine-Tuning` · `PEFT` · `MLflow` · `LoRA Adapter` · `Multi-Model Ensemble Labeling` · `Weak Supervision` · `Chain-of-Thought Structured Output` · `Citation Registry Constraint` · `Three-Tier Evaluation` · `LLM-as-Judge` · `Human Spot-Check Calibration`
- **Base Models**: `Qwen2.5-7B-Instruct` · `Qwen2.5-72B-Instruct` · `Claude 3.5 Sonnet` · `Claude 3.5 Haiku` · `GPT-4o` · `GPT-4o-mini`
- **PDF / Layout**: `Marker (ViT layout)` · `PaddleOCR + Spatial Clustering` · `Typhoon (Thai-specialized)` · `LlamaParse (fallback)` · `Citation Registry Extraction`
- **Deployment**: `Caravan` · `AWS SageMaker (endpoint hosting)` · `Streamlit (chatbot UI)` · `caravan-rpc SDK (provide / client)`
