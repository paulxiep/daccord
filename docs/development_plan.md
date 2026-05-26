# D'accord — Development Sequencing Plan

## Context

D'accord ([README.md](../README.md)) is a private QLoRA fine-tune of Qwen3-8B for cross-jurisdiction privacy regulation mapping (SEA-4 + EU-spine + UK + DE + FR — 8 framework families). The architecture is already decided in the internal architecture plan (Pillar B). **Architecture is not in scope for this document — sequencing, parallelism, gates, risks, and tooling are.**

### Scope this cycle (Pillar B)

- **No Caravan integration.** Deploy via boto3-direct to SageMaker.
- **Pillar A (Caravan emitter)** deferred. **Pillar C (AgentCore-orchestrated compliance research agent on top of d-accord)** is a future phase, not in this scope but not foreclosed — d-accord's deployed endpoint is the tool Pillar C will consume.
- RAG / embeddings / semantic search and multi-modal FM patterns are exercised in sibling projects; d-accord does not redo those.

### Technical goals d-accord closes

- **LoRA/QLoRA fine-tuning** of a 7B base on consumer GPU (RTX 5080, 16 GB VRAM)
- **MLOps** — MLflow tracking, model registry, experiment artifacts
- **Regulated-industry narrative** — data privacy cross-jurisdiction as the substantive domain
- **SageMaker hosting** of the QLoRA adapter on a real-time endpoint via boto3
- **Hybrid serving with provenance tagging** — retrieval-first + QLoRA fallback at the endpoint; per-response `gold-retrieval` / `fine-tune-generalization` tagging
- **Cost discipline & operational excellence** — teardown scripts, cost ceilings, IAM least-privilege

### Out of scope (deliberate concessions)

- Customer-facing thought leadership · FM evaluation as a separate artifact · Secure private-network AI (VPC/PrivateLink/KMS).
- d-accord still produces an eval CSV with per-jurisdiction breakdown — surfaced as operational rigor, not as the headline deliverable.

### Landscape context (for scoping awareness, not positioning)

- **OneTrust DataGuidance** — 300+ jurisdiction regulatory research SaaS with AI Copilot (RAG-over-frontier). Adjacent, not competitor; d-accord is a methodology + model artifact, not a SaaS.
- **Harvey AI** — broad legal AI ($11B val); frontier-model approach; citation hallucination remains a known weakness on niche SEA regs.
- **SaulLM-7B** — closest methodological cousin (7B legal specialist, pretrained on 19M docs). Different task; useful prior-art reference.

**Effort baseline**: ~2 wk (~0.4–0.6 EM) under the Bedrock-batch M2 re-plan (~$3.25 Phase 1 + $50–100 Phase 2 SageMaker). Original plan was ~3 wk under the free-tier-only assumption; the ~1-week compression comes from (a) F9 strong-mid auto-label ensemble cutting hand-validation labor from ~25 h to ~10 h, (b) overnight batch jobs removing the local-dispatcher constraint, and (c) AWS account pulled forward to M2 so Phase 2 cuts to ~2 days when triggered.

### Phased execution

- **Phase 1 — Local validation + cloud-batch M2 ensemble (M0–M4)**: full data pipeline + QLoRA training on RTX 5080 + three-tier eval. M2 ensemble auto-labels via async Bedrock + Google AI Studio batch jobs (~$3.13 spend, overnight); AWS account is stood up at M2 (tier 6C, pulling 14A/14B forward from M5) so the SageMaker work at M5 is partial-warm. **All deliverables except the SageMaker endpoint close here.** Total Phase 1 paid-API spend: **~$3.25**.
- **Phase 2 — SageMaker hosting (M5)**: deploy adapter to endpoint via boto3 in `ap-southeast-1`, smoke test, capture, tear down. Decoupled from Phase 1 timing; can be triggered when a concrete demo opportunity justifies the $50–100 SageMaker spend. AWS account already scoped at M2 → Phase 2 only needs teardown scripts + S3 model packaging + endpoint stand-up.
- **Phase C — AgentCore agent (future, separate repo)**: consumes d-accord's deployed endpoint as a tool. Out of scope here.

---

## 1. Tooling Decision — Training Framework

| Concern | Recommendation |
|---|---|
| Language | **Python** — no realistic alternative for QLoRA-7B; HF/PEFT/bitsandbytes is the ecosystem |
| Framework | **`transformers` + `peft` + `bitsandbytes` + `trl` + `accelerate`** as the default |
| Speed/VRAM fallback | **Unsloth** if M3 small-sweep OOMs or trains >2× slower than expected. Drop-in over HF + PEFT; fused Triton kernels for LoRA forward/backward + FlashAttention-2 tuned for LoRA → ~2× throughput, ~40–50% lower peak VRAM. Qwen3-8B supported. Trade-off: lags on newest models; nf4 4-bit only |
| Avoid | Axolotl / LLaMA-Factory — YAML-config wrappers. Engineering choices (LoRA rank, target modules, LR schedule, collator) live in config not Python, so an external reader sees "ran the wrapper" not "made deliberate choices". Fine in a production team; wrong when the deliberate choices are part of the deliverable |
| Experiment tracking | **MLflow autolog** wired from the toy run forward, not added later |

---

## 2. Task Graph

**Notation**: tasks are numbered by tier. **Numbers are ordinal — tier *N+1* cannot start until tier *N* closes.** **Letters within a tier are parallel — 1A, 1B, 1C run concurrently.** Gates `[M*]` mark milestone checkpoints; see §4 for DoD and cut criteria.

### Phase 1 — Local validation + cloud-batch M2 ensemble

| Tier | Tasks | Type | Notes |
|---|---|---|---|
| **1** | 1A repo skeleton + lockfile · 1B MLflow + autolog plumbing · 1C per-provider RPD caps for free-tier APIs (Groq/Cerebras/Google AI Studio/DeepSeek) · 1D start PDF corpus download | parallel | All independent; day-1 |
| **2** | 2A 20-pair hand-built toy gold · 2B eval harness (citation match + judge) · 2C tokenizer audit (Thai/FR/DE) · 2D Thai parser bake-off on 5-page sample (Marker vs Typhoon-OCR — **Marker locked**, see `data/parser_bakeoff/summary.md`) | parallel | No full corpus needed |
| **3** | 3A baselines on toy gold (base Qwen 3-8B + Llama 4 Scout via Groq + Gemini 3.1 Flash Lite via Google AI Studio) · 3B lock parser choice from 2D results | parallel | 3A needs 2A+2B; 3B needs 2D; baselines are all free-tier OSS to match the tier 7A ensemble |
|  | **[M0 gate]** | | tokenizer passes · baselines captured · parser locked |
| **4** | Parse all PDFs to markdown (Marker, locked for both EN and TH) | sequential | Needs 1D complete + 3B parser choice. **Watch R8**: 3 sources (UK-GDPR, UK DPA 2018, FR Loi I+L) come from browser print-to-PDF (Légifrance/legislation.gov.uk expose no scraper-friendly consolidated PDF) — 5–60× larger than regulator-issued PDFs, layout may confuse Marker |
| **5** | Citation registry extraction per framework | sequential | |
|  | **[M1 gate]** | | corpus + registries frozen |
| **6** | 6A ensemble prompt + JSON schema (citations constrained to registry from 5) · 6B tiering script · **6C AWS preliminary (tier 14A/B pulled forward from M5)**: IAM user `d-accord-dev` + scoped S3 bucket `s3://daccord-dev-{account_id}/` in **`ap-southeast-1`** + AWS Budgets alarm ($50/$100) + Bedrock model-access requests for the 4 chosen ensemble models · 6D Google AI Studio billing enable + `GEMINI_PAID_API_KEY` provisioning | parallel | 6A/6B are code; 6C/6D are account setup. All four can run in parallel on d7. |
| **7** | 7A ensemble generation — **4-seat F9 ensemble via async batch APIs**: Llama 4 Scout (Bedrock batch) + Llama 4 Maverick (Bedrock batch) + Claude Haiku 4.5 (Bedrock batch) + Gemini 3.1 Flash (Google AI Studio batch). Submit 4 batch jobs at evening of d7, results back morning of d8 (typical 1–12 h, 24 h SLA). · 7B splits script · 7C hand-validate completed framework-pairs as they land | parallel | 7A is fire-and-forget overnight — no local dispatcher running for hours. F9 strong-mid models reduce expected MED/LOW hand-val rate to ~10% (vs ~25% on cheap free-tier ensemble), cutting 7C labor from ~25 h to ~10 h. |
| **8** | Tiering (HIGH/MED/LOW/SALVAGE) + complete hand-validation + HIGH-tier per-jurisdiction spot-check | sequential | Needs 7A complete + all 7C |
| **9** | Gold freeze (≥500 pairs) + jurisdiction-disjoint train/val/test splits + dataset SHA | sequential | |
|  | **[M2 gate]** | | gold + splits frozen with version hash; AWS account scoped + Bedrock access provisioned |
| **10** | 10A `training/train.py` (HF `transformers` + `peft` + `bitsandbytes` + `trl`) · 10B small-sweep config | parallel | 10A can start during the 7A overnight window |
| **11** | Small-sweep — 200 pairs × 1 epoch | sequential | Validates MLflow plumbing, adapter save/reload, OOM headroom |
|  | **[M3 gate]** | | adapter saves/reloads · MLflow logs run + SHA · no OOM at target seq_len (else swap to Unsloth) |
| **12** | 12A full QLoRA train + small hyperparam sweep (~overnight **async**) · 12B three-tier eval script + retrieval baseline (MPNet+FAISS over train-split source clauses) + `build_retrieval_index.py` · 12C draft Phase 2 deploy/teardown scripts + hybrid inference handler (`publish/sagemaker_handler.py`) + Streamlit side-by-side app (`consumer/app.py`) | parallel | 12A async; 12B/12C fill the wait. 12C is now substantive (~2–3 d) — don't rush. |
| **13** | Three-tier eval across 4 comparators (fine-tune + base Qwen + Llama 70B + retrieval) run twice with `--slice-tag in-domain` and `--slice-tag out-of-domain`; per-jurisdiction + per-language breakdown aggregated from CSV rows. Eval-judge calls route through **Bedrock Haiku 4.5** (account already warm from M2) instead of free-tier Gemini — eliminates the tier 13 RPD bottleneck. | sequential | Slice tag goes to MLflow run metadata, not per-row (CSV contract stable per [eval/README.md](../eval/README.md)). ~$0.10 spend over 500-pair eval. |
|  | **[M4 gate]** | | Phase 1 done — eval CSV + MLflow history + adapter on disk |

### Phase 2 — SageMaker hosting (triggered separately)

Tier 14A (IAM user) + 14B (Budgets alarm) + Bedrock model access were pulled forward to M2 (see tier 6C above). Only 14C + 14D remain at Phase 2.

| Tier | Tasks | Type | Notes |
|---|---|---|---|
| **14** | 14C **teardown scripts committed before any stand-up** (`scripts/teardown_endpoint.py`, `scripts/teardown_all.py --nuke`) · 14D adapter + retrieval index + embedder snapshot + custom inference handler packaged to SageMaker S3 layout via `publish/package_model.py` | parallel | Stand-up (tier 15) blocked until 14C is in git. 14A/14B already done at M2. |
| **15** | SageMaker endpoint stand-up via boto3 (`ml.g5.xlarge`, `ap-southeast-1`) | sequential | ~5–10 min cold start |
| **16** | Smoke test 10 source clauses via side-by-side comparison view (5 in-domain, 5 out-of-domain); verify provenance tags (`gold-retrieval` / `fine-tune-generalization`) return correctly + CSV export round-trip | sequential | |
| **17** | Capture — recording + screenshots | sequential | |
| **18** | Endpoint teardown | sequential | Within 48 h of capture · spend <$100 |
|  | **[M5 gate]** | | recording captured · endpoint down · adapter remains in S3 |

---

## 3. Execution Notes

**The two long async jobs** are **7A** (ensemble generation, **overnight via async batch APIs**) and **12A** (full QLoRA train, overnight on local GPU). 1D (PDF download) is also unattended but short. Both 7A and 12A are submit-and-walk-away: 7A submits 4 batch jobs (3 to Bedrock in `ap-southeast-1`, 1 to Google AI Studio) at end of d7 and reads results back on d8 morning; 12A queues on the RTX 5080 at end of d11 and training completes overnight for the d12 eval pass.

**Ensemble checkpointing**: write `data/ensemble/raw/{framework_pair}__{model}.jsonl` as each batch job completes; the poll script (`scripts/run_ensemble.py --poll`) is idempotent and resumable — re-running picks up any unfinished jobs. Bedrock batch jobs survive operator-side interrupts because the work runs cloud-side.

**Rough timeline** (solo dev, baseline velocity):

| Days | Tiers | Closes | Notes |
|---|---|---|---|
| d1–3 | 1 + 2 + 3 | **M0** | unchanged |
| d4–6 | 4 + 5 | **M1** (end of week 1) | unchanged |
| d7 | 6A + 6B + **6C/6D AWS+Google account setup** | — | AWS prelim parallel with prompt/tiering code |
| d7 eve | 7A submit (4 batch jobs) + 7B splits + start 7C hand-val | — | submit and walk away |
| d8 morn | 7A poll/download (~1 h compute) + 8 tiering + finish 7C hand-val | — | F9 strong-mid ensemble → ~10 h total 7C labor (vs ~25 h on free-tier) |
| d8 eve | 9 gold freeze + splits + dataset SHA + MLflow tags | **M2** | **3 days post-M1** (was 4) |
| d9 | 10A train script + 10B sweep config | — | |
| d10 | 11 small-sweep (200 pairs × 1 epoch) | **M3** | 2 days post-M2 (unchanged) |
| d11 eve | 12A queue overnight train + 12B eval script + retrieval baseline + 12C deploy/Streamlit | — | parallel |
| d12 | 13 three-tier eval (Bedrock Haiku 4.5 judge) | **M4** — Phase 1 done | **d12 (was d15) — 3 days saved** |
| d13–14 | 14C teardown scripts + 14D S3 model packaging + 15 endpoint + 16 smoke + 17 capture + 18 teardown | **M5** | tier 14A/B already done at M2 → Phase 2 cuts to ~2 days |

---

## 4. Milestone Gates

### Phase 1 — Local validation + cloud-batch M2 ensemble (M0 → M4)

All ML substance happens here. AWS account stood up at M2 (tier 6C, pulling 14A/14B forward) for Bedrock batch ensemble + future SageMaker reuse; total Phase 1 paid-API spend ~$3.25 (M2 ensemble ~$3.13, M4 eval judge ~$0.10). If Phase 2 is deferred indefinitely, Phase 1 still constitutes a complete deliverable set (eval CSV + MLflow runs + local-inference recording + README + provisioned AWS account ready for re-use).

### M0 — Eval Bar Locked (end of d3)

- **DoD**: 20-pair toy gold built · eval harness runs end-to-end · baselines captured (base Qwen 3-8B + Llama 4 Scout via Groq + Gemini 3.1 Flash Lite via Google AI Studio) on toy · tokenizer audit committed
- **Artifact**: `eval/baseline_toy.csv` + `eval/tokenizer_audit.md`
- **Cut criterion**: tokenizer audit shows Qwen3-8B fragments Thai at >2 tokens/char average → escalate immediately: swap base (SeaLLM-v3, Typhoon-7B) or descope Thai. **Decide here, not at training time.**

### M1 — Corpus + Registry Frozen (end of week 1)

- **DoD**: All 8 framework families parsed to markdown · registries extracted · parser-choice rationale in README
- **Artifact**: `data/registry/*.json` per framework + parser bake-off score table
- **Cut criterion**: Thai bake-off has no clear winner OR both candidates fail on Royal Gazette amendments → drop Royal Gazette (keep PDPA-TH core only); document the cut. (**Resolved 2D**: Marker won with 48/48 perfect citation extraction on PDPA-TH original; cut not triggered.)

### M2 — Gold Set Frozen (~d8)

- **DoD**: ≥500 hand-validated gold pairs · ensemble outputs checkpointed · HIGH-tier stratified spot-check shows no jurisdiction <80% sample quality · jurisdiction-disjoint train/test split committed with dataset hash · **AWS account scoped + Bedrock model access provisioned for the 4 F9 ensemble members + Google AI Studio billing enabled** (tier 14A/14B pulled forward from M5)
- **Artifact**: `data/gold/gold_v1.jsonl` + `data/splits/{train,val,test}.jsonl` + spot-check report + `data/ensemble/raw/*.jsonl` (4 model outputs × ~30 framework-pairs each)
- **Ensemble**: F9 strong-mid auto-label-optimized — **Llama 4 Scout** (Bedrock, 2025-04) + **Llama 4 Maverick** (Bedrock, 2025-04) + **Claude Haiku 4.5** (Bedrock, 2025-10) + **Gemini 3.1 Flash** (Google AI Studio, 2026-01). All capability-balanced strong-mid tier; tier 8 HIGH=4/4 agreement logic preserved. ~$3.13 spend at mid scope (~30 framework-pairs, ~12 K total raw candidates, ~3 K per seat).
- **Cut criterion**: gold <300 by d8 → drop Malaysia + Philippines (cheap completers); two-native-language story stays intact. Secondary cut: any single Bedrock model-access request still pending by d7 morning → fall back to a 3-seat Bedrock-only ensemble (drop Gemini, retier tier 8 to HIGH=3/3).

### M3 — Small-Sweep Validated (~d10)

- **DoD**: 1 epoch × 200 pairs trains end-to-end · loss curve sensible · adapter saves+reloads cleanly · MLflow autolog shows the run with adapter SHA logged
- **Artifact**: MLflow screenshot + sanity-check inference output
- **Cut criterion**: OOM at QLoRA-7B on the 5080 → drop max_seq_len 4096→2048, add gradient checkpointing, micro-batch 1 + grad-accum 16. If still OOM, **swap to Unsloth** before full train.

### M4 — Eval Delta Proven (~d12) — Phase 1 done

- **DoD**: Three-tier eval against M2 gold · per-jurisdiction + per-language breakdown · delta vs M0 baselines numerically captured
- **Artifact**: `eval/results_v1.csv` + per-jurisdiction breakdown table; MLflow run history; updated README
- **Eval-judge routing**: Bedrock Haiku 4.5 (account already warm from M2) instead of free-tier Gemini — eliminates tier 13's prior 500-RPD bottleneck. ~$0.10 spend over 500-pair × 4-comparator eval.
- **Cut criterion**: fine-tune delta vs base Qwen <5% on Tier-1 citation accuracy AND negative vs Llama 3.x 70B on every jurisdiction AND **no advantage over the retrieval baseline on the out-of-domain slice** → **do not push to SageMaker**. The retrieval-baseline qualifier is what makes the cut honest: if retrieval ties or beats fine-tune everywhere, ship as retrieval-only (architecture pivot), don't ship the heavier serving stack just to preserve the original framing. Document the honest negative result in the eval CSV. (Total Phase 1 paid-API spend if cut: ~$3.25 — the M2 ensemble batch already submitted, the M4 eval judge is post-train so can be skipped if cut triggers earlier.)

### Phase 2 — SageMaker hosting (M5)

Trigger when M4 has a publishable delta AND there's a concrete reason (demo, runway) to absorb the AWS spend. AWS account is already warm from M2 (IAM, S3, Budgets all done at tier 6C) so cold-start time is just teardown-scripts + S3 model packaging + endpoint stand-up (~2 days).

### M5 — Endpoint Live, Captured, Torn Down (~2 days when triggered)

- **DoD**: Endpoint live · side-by-side comparison view returns retrieval + fine-tune + base outputs with provenance tags for 5 test source clauses · CSV export verified · short screen recording captured · **endpoint torn down** · S3 artifact remains (cheap)
- **Artifact**: Recording + 4–6 screenshots + adapter S3 URI
- **Cut criterion**: endpoint burn rate puts $100 ceiling at risk → tear down within 48 h of capture. **The durable artifact is the recording, not the running endpoint.** Re-stand-up on demand from `scripts/deploy_endpoint.py` (~5–10 min cold start) when needed for a live demo.

---

## 5. LLM Fine-Tuning Practices to Layer In

- **Baseline-before-fine-tune (M0)** — non-negotiable. The "specialist achieves citation-faithful structural mapping with native-language validation moats" claim requires numerical proof against base Qwen 3-8B *and* strong OSS comparators (Llama 4 Scout via Groq, Gemini 3.1 Flash Lite via Google AI Studio) on the same eval set. No baseline → no defensible claim.
- **Tokenizer audit (M0)** — minutes to run. The chosen base's tokenizer must handle Thai/FR/DE without excessive fragmentation; verify empirically. Bytefallback >20% on Thai = hard stop. (Run 2026-05-25 on Qwen3-8B: PASS for th/fr/de/en.)
- **Small-sweep before full train (M3)** — 200 pairs, 1 epoch, ~30 min. Validates adapter save/reload, MLflow autolog capture, loss-curve shape, OOM behavior at full seq_len. Cheaper to discover plumbing breakage on 200 pairs than 5000.
- **MLflow autolog from the toy run** — log every run from day-1, including failed/aborted ones. A full run history is itself an artifact.
- **Per-jurisdiction breakdowns in metrics** — every eval row is `(jurisdiction_source, jurisdiction_target, citation_match, judge_score)`. Aggregate per-jurisdiction-pair in the CSV.
- **Jurisdiction-disjoint test slice** — hold out specific control areas (e.g., breach notification across jurisdictions in test; data subject rights in train). Detects overfitting to specific control families.
- **Reproducibility**: pin `torch`/`transformers`/`peft`/`bitsandbytes`/`trl` versions in a lockfile; set seeds (`torch`, `numpy`, `random`, `transformers.set_seed`); log adapter SHA256 + git commit hash in MLflow params; hash and version the gold dataset; `eval/results_v1.csv` references the hash explicitly.

---

## 6. Cloud / Cost Practices

- **Phase 1 spend** is **~$3.25** — M2 ensemble batch (~$3.13: F9 ensemble across Bedrock + Google AI Studio) + M4 eval judge (~$0.10: Bedrock Haiku 4.5 over 500-pair eval). Plus ~$5–10 LlamaParse fallback if Marker fails on a specific document. Free-tier providers (Groq, Cerebras, DeepSeek) are no longer in the critical path — the prior plan's free-tier RPD pacing was the source of the 3-day async wait at 7A. One row per day in `costs/daily.csv` committed to repo with provider-specific entries; F9 batch invocations log against `bedrock_batch` and `gemini_paid` provider keys with their pre-priced rates in [costs/config.toml](../costs/config.toml). Hard $5/day USD cap per provider enforces the ceiling.
- **Phase 2 SageMaker discipline**: `ml.g5.xlarge` ≈ $1.40/hr in `ap-southeast-1`; **target <48 h total live**; stand up → smoke test (10 prompts) → capture → tear down (~2 h live). Re-stand-up on demand from `scripts/deploy_endpoint.py`; budget for ~5–10 min cold start for live demos. Cold start now also loads the MPNet embedder + FAISS index alongside the 7B adapter (~1–2 GB additional read; negligible time impact vs the adapter load).
- **AWS account scope** (pulled forward from M5 to M2 prelim at tier 6C): IAM user `d-accord-dev`, never root; policies scoped to `s3:* on arn:aws:s3:::daccord-dev-{account_id}/*` (M2 artifacts) + `bedrock:InvokeModel*` + `bedrock:CreateModelInvocationJob*` on the 4 F9 models; `sagemaker:*` on resources tagged `Project=d-accord` added at M5. Region: **`ap-southeast-1`** (Singapore) for both Bedrock and SageMaker.
- **S3 versioning** enabled on `daccord-dev-{account_id}` (trivial cost, prevents adapter clobber + protects M2 ensemble outputs from re-run overwrites).
- **Teardown as committed code** before first SageMaker stand-up (`scripts/teardown_endpoint.py`, `scripts/teardown_all.py --nuke`). Bedrock batch jobs are on-demand (no persistent endpoint) so M2 needs no teardown beyond cost-tracker reconciliation.
- **API spend resilience**: ensemble outputs checkpointed per `(framework_pair, model)` to `data/ensemble/raw/`; batch poll script (`scripts/run_ensemble.py --poll`) is idempotent and resumes if interrupted. Bedrock batch jobs run cloud-side and survive operator-side interrupts.
- **Project tag** `Project=d-accord` on every AWS resource (S3 bucket, Bedrock batch jobs, SageMaker endpoint, IAM policies) for cost attribution.
- **AWS Budgets alarm**: $50 warning + $100 hard threshold scoped to the `Project=d-accord` cost-allocation tag — established at tier 6C, applies across M2 and M5.

---

## 7. Risk Register

| # | Risk | Likelihood | Impact | Mitigation | Early-warning signal |
|---|---|---|---|---|---|
| R1 | Thai parser bake-off has no clear winner; registries unreliable | Medium | High (kills SEA differentiation) | Bake-off in M0/M1 on 5-page sample; cut = drop Royal Gazette, keep core PDPA-TH | ~~Day-2 bake-off scores cluster within 5% across all 3 parsers~~ **Resolved 2D**: Marker locked (48/48 citation extraction, ~2× faster than Typhoon; PaddleOCR candidate dropped pre-execution). |
| R2 | Ensemble agreement collapses on SEA frameworks (shared blind spots) | Medium-High | High (gold set thin) | Stratified human spot-check on HIGH-tier *per jurisdiction* (M2); cut = drop weakest 2 SEA jurisdictions | HIGH-tier spot-check <70% on any one jurisdiction |
| R3 | Gold dataset stalls <500 pairs by M2 | Medium | Medium (eval power weakens) | Cut to 6 jurisdictions (drop MY + PH); reuse HIGH-tier with 10% audit as proxy | By d8 morning (after 7A batch results land), HIGH-tier yield projects <400 pairs at full hand-val |
| R4 | Qwen3-8B tokenizer fragments Thai/FR worse than expected | Low-Medium | High (kills language-validation moat) | M0 tokenizer audit before training plumbing; swap to SeaLLM-v3 or Typhoon-7B if Thai byte-fallback >20% | Audit shows >2.5 tokens/char on Thai or byte-fallback artifacts |
| R5 | RTX 5080 OOM at QLoRA-7B with full seq_len | Medium | Medium (slows training) | M3 small-sweep catches before full train; mitigations: max_seq_len 2048, gradient checkpointing, micro-batch 1 + grad-accum 16; **swap to Unsloth** if needed | Small-sweep OOMs at any seq_len >1024 |
| R6 | Fine-tune delta vs frontier baseline marginal or negative | Medium | High (kills the headline value claim — "small specialist beats / matches frontier") | M0 baselines set expectation early; M4 cut = document honest negative result and skip Phase 2 | Tier-1 citation accuracy delta vs base Qwen <5% by mid-train checkpoint |
| R7 | Bedrock model-access provisioning or Google AI Studio billing setup delays 7A submission past d7 evening | Low-Medium | Low (1-day slip, not $ ceiling breach) | All 4 F9 models requested in parallel at d7 morning under tier 6C; if any single model is still pending by d7 evening, **fall back to 3-seat Bedrock-only ensemble** (drop Gemini; retier tier 8 HIGH=3/3 with documented impact); hard $5/day USD cap per provider in [costs/config.toml](../costs/config.toml) guards spend ceiling. Batch jobs run cloud-side so operator-side interrupts are free. | Any Bedrock `bedrock:ListFoundationModels` call at d7 morning returns `modelLifecycle.status != ACTIVE` for a chosen model OR Google AI Studio billing enable still pending |
| R8 | Browser-print PDFs (UK-GDPR, UK DPA 2018, FR Loi I+L) parse noisily under Marker — Légifrance + legislation.gov.uk expose no scraper-friendly consolidated PDF, so 1D fell back to print-to-PDF (5–60× larger than regulator-issued PDFs, embedded page chrome) | Medium | Medium (registry drift on EU-spine + UK) | At tier 4, spot-check Marker output on these 3 vs a regulator-issued reference (e.g., BDSG); fallback = headless-browser PDF export with print-CSS suppression, or source from Legifrance API / legislation.gov.uk Atom feed | Tier 4 Marker output for UK/FR contains >2× the unrecognised tokens or broken citation IDs vs auto-downloaded sources |
| R9 | Retrieval baseline dominates fine-tune on in-domain AND ties on out-of-domain → architectural claim (specialist value-add) weakens | Medium | Medium (reframes the value claim, doesn't kill it) | Ship the eval CSV honestly; reframe README from "fine-tune is the engine" to "hybrid serving with provenance — retrieval for known, fine-tune for novel" (current default framing already supports this). If retrieval wins everywhere, ship as retrieval-only and demote the QLoRA adapter to "experimental comparator" in the project narrative — MLflow runs + training script + adapter SHA still constitute a real technique deliverable. | M3 small-sweep loss curve plateaus near base-Qwen on validation; OR M4 eval shows retrieval >= fine-tune on out-of-domain slice |

---

## 8. Verification

End-to-end DoD for this development cycle:

**Phase 1**:
1. M0–M4 all hit (or their cut criteria triggered with documented decisions)
2. `eval/results_v1.csv` shipped with per-jurisdiction breakdown
3. MLflow run history populated with adapter SHA + dataset hash linkage
4. README updated with parser-bakeoff rationale and eval results
5. Reproducibility: locked Python env · seeded train script · dataset hash referenced in `eval/results_v1.csv`
6. **AWS account scoped at M2**: IAM user `d-accord-dev` exists · S3 bucket `daccord-dev-{account_id}` versioned in `ap-southeast-1` · AWS Budgets alarm fired (test alert) at $50/$100 thresholds · Bedrock model access ACTIVE for all 4 F9 ensemble models
7. **M2 ensemble cost reconciliation**: `costs/daily.csv` rows for `bedrock_batch` and `gemini_paid` providers sum to <$5 per day per provider (USD cap held)

**Phase 2 (when triggered)**:
8. Adapter packaged in SageMaker-compatible S3 layout · adapter S3 URI documented
9. Endpoint deploy + teardown scripts committed and tested · `Project=d-accord` tag spend remains <$100 across M2 + M5 combined
10. Endpoint live for capture session · recording + screenshots in repo · endpoint torn down within 48 h of capture
11. SageMaker-scoped IAM policy JSON committed (extends the M2 baseline policy with `sagemaker:*` on `Project=d-accord`-tagged resources); S3 versioning verified

---

## 9. Tier 1–3 Implementation Detail

Sections 2–4 describe sequencing at the tier-row granularity. This section zooms in on tiers 1–3 specifically: what code actually shipped per task letter (with file pointers), where the remaining gaps are, and the concrete plan to close them. A returning contributor should be able to read §9 alone and know (a) the current implementation surface, (b) how to close the two outstanding 1–2 gaps, and (c) how to execute tier 3.

### Status snapshot

| Task | Status | Notes |
|---|---|---|
| 1A repo skeleton + lockfile | ✓ Done | See §9.1 |
| 1B MLflow plumbing | ✓ Done (manual; framework `autolog` deferred to 10A) | See §9.1 |
| 1C per-provider RPD caps | ✓ Done (2026-05-25) — Groq + Gemini + Cerebras + DeepSeek all wired with verified quotas | See §9.1 |
| 1D PDF corpus download | ✓ Done — 13 PDFs across 8 jurisdictions | See §9.1 |
| 2A 20-pair toy gold | ⚠ **9/20 author-verified, 11/20 claude-extract-only pending paraphrase pass, 0 STUBs remaining** (2026-05-25) — `data/gold/toy_v1.jsonl` is schema-valid; STUB pass complete (7 confirmed, 2 repointed to a different target framework — toy_003 FR→SG, toy_016 FR→TH). FR coverage dropped 4→2 (below ≥3 target). See [data/gold/toy_v1_provenance.md](../data/gold/toy_v1_provenance.md) for the full audit trail. | **Decision (2026-05-25): M0 closes on partial gold.** STUB pass + repoint decisions complete; semantic verification of the 11 claude-extract-only rows + FR coverage decision deferred to next MR. See §9.2 + §9.5 |
| 2B eval harness | ✓ Done — citation match + Gemini judge; tested end-to-end | See §9.2 |
| 2C tokenizer audit | ✓ Done & PASS for th/fr/de/en — R4 resolved | See §9.2 |
| 2D parser bake-off | ✓ Done — Marker locked; R1 resolved | See §9.2 |
| 3A baselines on toy gold | ⚠ Partial (this MR; runs against partial-verified gold per 2A deferral) — `envs/baseline/` + `LocalHFClient` + `GroqJudge` shipped; `qwen` / `groq` / `qwen3` / `gemini` aliases all wired. `eval/baseline_toy.csv` has 80 rows (4 generators × 20 pairs). Base swapped mid-session from Qwen 2.5-7B to Qwen 3-8B (newer multilingual tokenizer; re-audit on Qwen3-8B passed for all 4 languages); judge bumped Llama 3.3-70B → Llama 4 Scout for stronger signal. | Plan in §9.3 |
| 3B lock parser choice (README) | ⏳ In progress (this MR) — numeric outcome lands in README parser line | Plan in §9.3 |

### 9.1 — Tier 1 detail (what landed + 1C closure plan)

**1A — Repo skeleton + lockfile.** Layered uv-project layout, all gitignored venvs:

- **Root project** (`pyproject.toml` + `uv.lock` + `.venv/`) — pure-Python shared library + lightweight CLI scripts. Core deps: `pydantic>=2.13`, `mlflow>=2.18`, `httpx>=0.28`, `pyyaml>=6`. Dev tooling: `ruff` (lint + format, line-length 100), `pyright` (basic mode), `pytest`.
- **`src/daccord/`** — shared library; layout reflects pipeline stages:
  - [validation.py](../src/daccord/validation.py) — `@validated` decorator + `ValidatedModel` base class (pydantic `validate_call` wrapper); applied to every named function/method/class per the project's typed-validation convention.
  - [tracking.py](../src/daccord/tracking.py) — MLflow plumbing (see 1B).
  - [costs/](../src/daccord/costs/) — cost tracker (see 1C).
  - [corpus/](../src/daccord/corpus/) — PDF downloader (see 1D).
  - [eval/](../src/daccord/eval/) — eval harness library (see 2B).
  - [gold/schema.py](../src/daccord/gold/schema.py) — `GoldPair` + `GoldSet` (see 2A).
  - [bakeoff/](../src/daccord/bakeoff/) — Thai parser bake-off (see 2D).
  - [tokenizer_audit.py](../src/daccord/tokenizer_audit.py) — language-fragmentation audit (see 2C); dependency-injected encoder/decoder so root env doesn't pull in `transformers`.
- **`envs/<tier>/` sub-projects** — each is its own uv project (independent `pyproject.toml`, `uv.lock`, `.venv/`) linked to root `daccord` via path source. Currently: `envs/eval/` (2B), `envs/audit/` (2C), `envs/bakeoff/` (2D). Tier 3A adds `envs/baseline/`; tier 10A adds `envs/training/`. Rationale: bakeoff's marker-pdf / typhoon-ocr line and audit's `transformers v5` line have incompatible resolver constraints and must lock independently.

**1B — MLflow plumbing.** Wired manually rather than via framework autolog (the latter belongs in the training script at tier 10A):

- [src/daccord/tracking.py](../src/daccord/tracking.py) — `setup_mlflow()` resolves tracking URI (env var → `file:./mlruns`), sets experiment; `log_standard_params()` logs the reproducibility contract (`run_name`, `git_commit`, `seed`, `dataset_hash`, extras) and tags `project=d-accord`; `log_adapter_sha256()` for tier 10–13; `set_all_seeds()` seeds `random` + `numpy` + `torch` + `transformers` (lazy-imported).
- Smoke test: [scripts/mlflow_smoke_test.py](../scripts/mlflow_smoke_test.py) exercises the plumbing end-to-end (URI resolution, run creation, param logging) and can be re-run anytime to validate the local MLflow store. Local backend at `mlruns/` (gitignored); Phase 2 swaps via `MLFLOW_TRACKING_URI` with no code change.
- Reuse: [src/daccord/eval/runner.py](../src/daccord/eval/runner.py) calls `setup_mlflow` + `log_standard_params` per parent run, nests one child run per generator, and logs metrics via `mlflow.log_metric()`. Tier 10A will add `mlflow.<flavor>.autolog()` at the top of `training/train.py`; the manual contract from 1B continues alongside.

**1C — Per-provider RPD caps.** ✓ Closed 2026-05-25. Cost tracker covers all 4 intended free-tier providers.

What's wired today:

- [costs/config.toml](../costs/config.toml) — committed for audit trail. `[caps_requests_per_day]` now lists `groq = 14400`, `google_gemini = 1500`, `cerebras = 1000`, `deepseek = 1000`. Cerebras cap derived from the 1M-tokens/day published [Cerebras free-tier limits](https://inference-docs.cerebras.ai/support/rate-limits) (the 5 RPM ceiling is the real bottleneck for tier 7A pacing); DeepSeek is a 5M-token signup credit (one-time, not recurring) — RPD here is a budgetary aid only, tier 7A must monitor cumulative token spend separately. Paid-spill fallback (`anthropic`, `openai`, `together`) remains under `[caps_usd_per_day]`, unused in Phase 1.
- [src/daccord/costs/](../src/daccord/costs/) — `config.py` (loads TOML, `Provider` Literal, `kind_of(provider)` classifies free_tier vs paid), `tracker.py` (`preflight()` raises `CapExceeded` before a call; `record_call()` logs + re-checks), `storage.py` (SQLite WAL inflight ledger, thread-safe), `cli.py` (`python -m daccord.costs status` / `rollup`).
- Override env vars: `DACCORD_COSTS_OVERRIDE=1` (bypass caps; e.g. for testing), plus path overrides for repo root / config / inflight DB / daily CSV.
- Wired into [src/daccord/eval/clients.py](../src/daccord/eval/clients.py) — every `GroqClient.generate` and `GeminiClient.generate` call routes through `preflight` (pre-call RPD check using a `chars/4` heuristic) and `record_call` (post-call with actual SDK-reported token counts).

**1C completion plan — ✓ DONE 2026-05-25.** (Steps below preserved as historical record; CerebrasClient + DeepSeekClient *adapters* remain tier-7A scope.):

1. **Verify free-tier quotas at run time** — Cerebras Cloud and DeepSeek publish quotas separately from their docs landing pages and revise frequently. Look up the current published numbers, record the URL + checked-on date in the commit message. If a provider lacks a clear RPD figure (Cerebras historically advertises tokens/min ceilings rather than raw RPD), set a conservative cap (e.g. 1000) and document the rationale inline in `costs/config.toml`.
2. **Extend the `Provider` Literal** in [src/daccord/costs/config.py](../src/daccord/costs/config.py) from `("anthropic", "openai", "together", "groq", "google_gemini")` to add `("cerebras", "deepseek")`. Both classified as `free_tier` by `kind_of(provider)` — re-read that method to confirm classification is enum-driven vs table-driven and add the entry in the right branch; do not accidentally add to the paid set.
3. **Add `[caps_requests_per_day]` entries** in [costs/config.toml](../costs/config.toml) — `cerebras = <verified>` and `deepseek = <verified>`. No `[pricing.*]` table needed; free-tier `estimate_cost` returns 0.0.
4. **Extend cost-tracker tests** to cover the two new providers — at minimum a `preflight` check that raises `CapExceeded` after the configured RPD ceiling, plus a `record_call` happy path. Live alongside existing costs tests.
5. **NOT in scope**: writing `CerebrasClient` / `DeepSeekClient` adapters in `src/daccord/eval/clients.py`. Those land with tier 7A. This step closes the *config* gap so 7A can enforce caps from call one.
6. **Verification**: `docker compose run --rm root uv run pyright` (catches Literal extension misses), `docker compose run --rm root uv run pytest` (covers new cost tests), `docker compose run --rm root uv run python -m daccord.costs status` (prints all four free-tier providers with `0/<cap>` usage on day-zero).

**1D — PDF corpus.** Idempotent download script + manifest; 13 PDFs across 8 jurisdictions on disk.

- Script: [scripts/download_corpus.py](../scripts/download_corpus.py) — flags `--sources`, `--raw-root`, `--manifest`, `--frameworks`, `--dry-run`; skips files whose SHA256 already matches the manifest. Supports URL-fetch (httpx) and manual placement (prints expected path + waits).
- Spec: [data/sources.yaml](../data/sources.yaml) — 13 source entries across 8 frameworks. 9 auto-fetch (EU GDPR, BDSG DE+EN, PDPA-TH English, PH DPA + IRR, PDPA-MY Act + Amendment, FR Loi I+L), 4 manual (UK-GDPR, UK DPA 2018, PDPA-SG, PDPA-TH Thai — per R8 the UK + FR sources fall back to browser-printed PDFs because legislation.gov.uk and Légifrance expose no scraper-friendly consolidated PDFs).
- Manifest: [data/raw_manifest.json](../data/raw_manifest.json) — list of `ManifestEntry(framework, jurisdiction, filename, sha256, fetched_at)`; auto-updated by the script.
- On-disk PDFs (13 files) under `data/raw/<jur>/<framework>/`: `de/bdsg/{bdsg_de_current,bdsg_en_current}.pdf`, `eu/gdpr/reg_2016_679_consolidated.pdf`, `fr/loi_il/loi_78_17_consolidated.pdf`, `my/pdpa_my/{pdpa_my_act709_bilingual,pdpa_my_amendment_act_a1727_2024}.pdf`, `ph/dpa_2012_ph/{dpa_2012_ph,dpa_2012_ph_irr_amended}.pdf`, `sg/pdpa_sg/pdpa_sg_current.pdf`, `th/pdpa_th/{pdpa_th_thai_2019,pdpa_th_english_2019}.pdf`, `uk/{dpa_2018,uk_gdpr}/*_current.pdf`.
- R8 follow-up at tier 4: spot-check Marker output on UK-GDPR / UK DPA 2018 / FR Loi I+L against a regulator-issued reference (e.g., BDSG); if unrecognised-token rate >2× the auto-downloaded baseline, switch to headless-browser PDF export with print-CSS suppression or source from Legifrance API / legislation.gov.uk Atom feed.

### 9.2 — Tier 2 detail (what landed + 2A creation plan)

**2A — 20-pair hand-built toy gold.** ⚠ **Drafted 2026-05-25; 9/20 author-verified; 11/20 claude-extract-only pending paraphrase semantic pass; 0 STUBs remaining.**

**Decision (2026-05-25): M0 closes on partially-verified gold.** Two phases of author verification happened this session:

1. **STUB pass — complete.** All 10 originally-STUB citation_ids (best-guess) were either author-confirmed (7: toy_002 BDSG Sec 38, toy_004 Loi I+L Art 121, toy_013/014/015 PDPA-TH มาตรา 19/37(4)/37(1), toy_017 UK DPA 2018 Sec 45, toy_018 UK DPA 2018 Sec 69 + BDSG Sec 5, toy_020 Loi I+L Art 49 + PDPA-TH มาตรา 30) or repointed to a different target framework (2: toy_003 FR→SG, toy_016 FR→TH — see provenance for the repoint rationale).
2. **claude-extract-only pass — deferred.** 11 rows (toy_001, 003 (repointed), 005, 006, 007, 008, 009, 010, 011, 012, 019) have citation_ids that claude extracted directly from the source PDFs — citation_id is correct, but the `*_mechanism` paraphrase has not been semantically verified by the author against the regulatory text. Next-MR task: spot-check or full pass.

**Outstanding verification debt (carried to next MR):**

- **11 claude-extract-only rows pending paraphrase semantic pass.** Citation_ids confirmed; risk is paraphrase drift from the regulator's actual wording. Lower risk than a STUB but technically still UNVERIFIED in the strict sense. Mitigation in the meantime: numbers in `eval/baseline_toy.csv` for these rows should be read as "structurally valid, semantically untested".
- **FR coverage at 2 pairs (toy_004, toy_020), below the ≥3 native-language-validation-moat target.** The toy_003 and toy_016 repoints both moved away from FR Loi I+L (003: no standalone French breach-notification provision exists — GDPR direct effect; 016: Art 8 was the wrong DPO article and the right one was not located in the post-2018 consolidated text). Decision deferred — next MR will either accept FR=2 with a documented gap, or repoint one of the claude-extract-only rows (toy_009 or toy_010 are the cheap candidates) to add an FR target.
- **dataset_hash changed during the STUB pass.** The baselines in `eval/baseline_toy.csv` are pinned to the *original* `dataset_hash = 412501438684f1ea…`; re-running against the current verified hash will produce a comparable but diff-able CSV (the verification delta).

See the banner at the top of [data/gold/toy_v1_provenance.md](../data/gold/toy_v1_provenance.md) and the per-pair log for the full audit trail.

What landed:

- Gold file: [data/gold/toy_v1.jsonl](../data/gold/toy_v1.jsonl) — 20 rows, schema-valid, `dataset_hash = 412501438684f1ea9c2fcfbdcbb92897cb469fd795c61f8839e193824e3880a5`. The hash will change once the author edits any cell during the verification pass.
- Coverage matrix: [data/gold/toy_v1_coverage.md](../data/gold/toy_v1_coverage.md) — pre-plan of the 20 pairs against jurisdiction × concept × difficulty axes. Coverage gates all PASS mechanically (20 rows, all 8 jurisdictions ≥2, th=4 ≥3, fr=4 ≥3) per [envs/eval/scripts/verify_toy_coverage.py](../envs/eval/scripts/verify_toy_coverage.py). Mechanical pass; does not validate citation correctness.
- Provenance + double-check punch list: [data/gold/toy_v1_provenance.md](../data/gold/toy_v1_provenance.md) — every row currently `verified_by: UNVERIFIED`. Two tiers of follow-up: 10 STUB rows (best-guess citation_id, likely needs correction) + 10 claude-extract-only rows (citation_id confirmed via direct PDF extract, mechanism text needs author semantic pass).
- Drafter (unused this round): `envs/eval/scripts/draft_toy_gold.py` calls Llama 4 Scout (Groq) + Gemini 3.1 Flash Lite (Google AI Studio) and writes `data/gold/.draft_*.jsonl` (gitignored). This round used direct PDF text extraction via [envs/audit/scripts/extract_pdf_text.py](../envs/audit/scripts/extract_pdf_text.py) instead, skipping the LLM-hallucination layer for the 6 frameworks claude could read first-hand (GDPR, PDPA-SG, PDPA-TH English, PDPA-MY, DPA-PH, BDSG English).

What's still needed before M0 can close (per the 2A acceptance criteria below):

1. Author opens each `data/raw/<jur>/<framework>/` PDF; for each pair confirms (a) citation_id exists with that exact string, (b) `*_mechanism` paraphrase is faithful to actual text.
2. STUB rows additionally need the citation_id corrected, not just confirmed (FR Loi I+L post-2018 renumbering, BDSG Sec 38 threshold, UK DPA 2018 Part 3 sections, etc.).
3. `verified_by` + `verification_date` cells in the provenance table get populated row-by-row.
4. PENDING notes in the JSONL's `notes` field get removed once the row is verified (changing `dataset_hash`).
5. Re-run [verify_toy_coverage.py](../envs/eval/scripts/verify_toy_coverage.py) + `run_eval.py --dry-run` to confirm nothing broke during the verification edits.

**If a future session sees weird `eval/baseline_toy.csv` numbers on a TH / FR / DE / UK pair — check the provenance banner first.** If the relevant pair is still flagged STUB, the gold is the bug, not the model under eval.

**2A verification plan (next MR, pre-tier-4)** — the per-row pass deferred from M0:

1. Re-confirm the coverage matrix in `data/gold/toy_v1_coverage.md` survived any cell edits during verification.
2. For every row, open the source PDF in `data/raw/<jurisdiction>/<framework>/` and:
   - Confirm the cited article/section exists in the PDF (exact `citation_id`, not paraphrased).
   - Confirm `target_mechanism` paraphrases the actual text (not the LLM's hallucinated gist).
   - For Thai PDPA pairs: cite from `pdpa_th_thai_2019.pdf` (authoritative) and cross-check `pdpa_th_english_2019.pdf`. Same pattern for BDSG (German authoritative).
   - For UK / FR pairs: these PDFs are browser-print (R8 caveat) — header chrome may shift page numbers; cite by article/section, never by page.
3. Fix STUB rows (FR Loi I+L post-2018 renumbering, BDSG Sec 38 threshold, UK DPA 2018 Part 3 sections, etc.).
4. Fill `verified_by` + `verification_date` in [data/gold/toy_v1_provenance.md](../data/gold/toy_v1_provenance.md) row-by-row; clear PENDING notes in the JSONL's `notes` field. Each edit changes the `dataset_hash`.
5. Schema-validate: `docker compose run --rm eval uv run python scripts/run_eval.py --dry-run --gold-path ../../data/gold/toy_v1.jsonl --verbose`. Loads via `GoldSet.from_jsonl`, prints the new `dataset_hash`, builds a sample prompt; zero API calls.
6. Re-run the §9.3 baseline command against the verified gold; diff `eval/baseline_toy.csv` against this MR's row-by-row to quantify verification impact.
7. **Acceptance criteria** (then unblocks tier 4):
   - [ ] `data/gold/toy_v1.jsonl` still contains exactly 20 lines; new `dataset_hash` recorded.
   - [ ] `--dry-run` loads cleanly.
   - [ ] All 8 jurisdictions appear ≥2 times (source or target); PDPA-TH and FR Loi I+L each appear ≥3 times.
   - [ ] Provenance table fully populated (no empty `verified_by` / `verification_date` cells).
   - [ ] `eval/baseline_toy.csv` re-run committed; per-row diff vs the pre-verification baseline summarised in the verification PR description.
8. **NOT in scope here**: scaling beyond 20 pairs (that's tier 9, gold freeze ≥500 pairs).

**2B — Eval harness.** Done; tested; runnable today.

- Sub-project: `envs/eval/` (own `pyproject.toml`, `uv.lock`, tests). Provider SDKs (Groq, Google GenAI) isolated here to keep the root venv slim.
- CLI: [envs/eval/scripts/run_eval.py](../envs/eval/scripts/run_eval.py). Flags: `--gold-path`, `--models` (comma-separated aliases — currently `qwen,groq,gemini,retrieval`), `--judge` (default `meta-llama/llama-4-maverick-17b-128e-instruct` via Groq), `--output-csv`, `--run-name`, `--prompt-variant`, `--seed`, `--dry-run`.
- Library: [src/daccord/eval/](../src/daccord/eval/).
  - `clients.py` — `ModelClient` Protocol (`provider: Provider`, `model: str`, `generate(messages, run_id, batch_id) → ModelResponse`); concrete `GroqClient` (JSON mode via `response_format={"type": "json_object"}`) and `GeminiClient` (native `response_json_schema`). Both lazy-import their SDK and route through the cost layer.
  - `scoring.py` — `normalize_citation_id` (strips "Article"/"Section"/whitespace, normalises "Sec." → "Section") + exact match for Tier-1; `JudgeClient` Protocol + `GeminiJudge` (continuous score [0,1] + bucket enum `wrong`/`partial_wrong`/`partial_right`/`substantively_right`/`exact` + reasoning).
  - `runner.py` — orchestrates generators + judge, emits CSV + MLflow runs.
  - `prompts.py`, `schema.py` — `PromptMessages`, `CitationCandidate`, `ModelResponse`.
- CSV row contract (14 columns, stable from M0 → M4): `gold_id, model, source_jurisdiction, source_framework, target_jurisdiction, target_framework, source_language, target_language, predicted_citation_id, expected_citation_id, citation_match, judge_score, judge_bucket, judge_reasoning`.
- MLflow shape: parent run (tags `project=d-accord`, `gate=M0`, `prompt_variant=unconstrained-m0`), one nested child per generator. Child metrics include `tier1_citation_match_overall`, per-jurisdiction (`tier1_citation_match__jur__<jur>`), per-language (`tier1_citation_match__lang__<lang>`), per-framework-pair (`tier1_citation_match__fwpair__<src>__<tgt>`), `tier2_judge_mean`, `tier2_judge_pct_above_0.7`, judge-bucket counts.
- Tests: `envs/eval/tests/test_eval_{runner,clients,prompts,schema,scoring}.py` — 40+ assertions covering end-to-end mocked generation + judging, CSV contract stability, MLflow nesting, aggregation.
- See also: [eval/README.md](../eval/README.md).

**2C — Tokenizer audit.** Done; executed 2026-05-24; verdict **PASS** for all four languages. R4 resolved.

| Lang | Source | tokens/char | single-byte frac | Verdict |
|---|---|---:|---:|---|
| th | `pdpa_th_thai_2019.pdf` (pp. 14–22) | 0.575 | 0.055 | PASS |
| fr | `loi_78_17_consolidated.pdf` (pp. 48–56) | 0.520 | 0.613 | PASS |
| de | `bdsg_de_current.pdf` (pp. 15–23) | 0.303 | 0.172 | PASS |
| en | `reg_2016_679_consolidated.pdf` (pp. 26–34) | 0.213 | 0.180 | PASS |

Methodology + diagnostic tables (incl. top-fragmented Thai characters) in [eval/tokenizer_audit.md](../eval/tokenizer_audit.md); machine-readable in [eval/tokenizer_audit.csv](../eval/tokenizer_audit.csv); logged to MLflow run `6537272d8e7f4af0a72929ec6573a93d` under experiment `daccord-tokenizer-audit`. Reproduce: `docker compose run --rm audit uv run python scripts/run_tokenizer_audit.py`. Qwen3-8B is locked as the base model.

**2D — Thai parser bake-off.** Done; **Marker locked**. R1 resolved.

5-page Thai sample (PDPA-TH 2019, pages [1, 11, 16, 39, 42]) scored against 61 hand-verified Thai citations (`มาตรา N`). Both parsers hit perfect recall + precision; reading-order is the tie-breaker.

| Parser | Recall | Precision | Reading order (Thai-reader judgment) | Structure preserved |
|---|---:|---:|---:|---:|
| marker | 1.000 | 1.000 | 5.000 | 1.000 |
| typhoon | 1.000 | 1.000 | 4.000 | 1.000 |

Full report: [data/parser_bakeoff/summary.md](../data/parser_bakeoff/summary.md). Decision applies to tier 4 (full-corpus parse to markdown) for both EN and TH. R1 cut criterion not triggered.

### 9.3 — Tier 3 detailed plan

**Prerequisite**: §9.2's 2A draft gold is committed (verification deferred to next MR; see banner). 3A runs against the current `dataset_hash = 412501438684f1ea…`. 3B is doc-only and parallelisable.

#### 3A — Baselines on toy gold

Run three generators on the 20-pair toy gold, judged by Llama 4 Scout via Groq, and emit `eval/baseline_toy.csv` + MLflow run history. The three baselines are: **base Qwen 3-8B (local, 4-bit NF4)** + **Llama 4 Scout via Groq** + **Gemini 3.1 Flash Lite via Google AI Studio**.

The Qwen baseline runs at **4-bit NF4 quantization** (the same precision the tier 10–12 QLoRA training will load at), not bf16 — this is a deliberate choice so the M4 fine-tune-delta is apples-to-apples with the actual production load condition, not against a stronger-than-shipped baseline.

1. **Scaffold `envs/baseline/` sub-project + `baseline` compose service.** Mirrors `envs/eval/` / `envs/audit/` / `envs/bakeoff/`:
   - [envs/baseline/pyproject.toml](../envs/baseline/pyproject.toml) — `requires-python = ">=3.14,<3.15"`; declares `daccord` via `[tool.uv.sources]` path source `../../`; pins `transformers>=5.9,<6` + `torch>=2.7,<3` + `bitsandbytes>=0.49,<1` + `accelerate>=1.2,<2` (versions match `consumer/pyproject.toml` so baseline + consumer load Qwen the same way).
   - 6th compose service `baseline` in [docker-compose.yml](../docker-compose.yml) — `Dockerfile.cuda` image, `working_dir: /workspace/envs/baseline`, `deploy.resources.reservations.devices` GPU passthrough (mirrors bakeoff), own `daccord-venv-baseline` named volume, `env_file: .env.local` for the Groq/Gemini API keys the eval runner needs.
   - `envs/baseline/.venv/` gitignored via existing `envs/*/.venv/` rule.
   - Sync: `docker compose run --rm baseline uv sync` (first run downloads torch + bnb wheels into the shared `daccord-uv-cache` volume — ~3 GB).
2. **Add `LocalHFClient` to [src/daccord/eval/clients.py](../src/daccord/eval/clients.py)** implementing the existing `ModelClient` Protocol:
   - `provider: Provider = "local_hf"` — added to the `Provider` Literal in [src/daccord/costs/config.py](../src/daccord/costs/config.py) following the existing `"retrieval"` precedent (local-only providers have no cap entry in `costs/config.toml` and bypass `preflight` / `record_call` because `daily.csv` is the spend log, not a generic call ledger). Latency + token counts still flow on `ModelResponse` → CSV + MLflow via the runner.
   - `__init__(model="Qwen/Qwen3-8B", quantization="nf4")` — lazy-import torch + transformers + bitsandbytes; load with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)`; cache model + tokenizer on instance.
   - `generate(...)` — apply the Qwen chat template; greedy decode (`do_sample=False`); reuse `_parse_candidate` for failure recovery (parse failures record as Tier-1 misses with the error in `judge_reasoning`). Qwen has no native JSON-schema constraint — output discipline is prompt-only.
   - Decorate every public method with `@validated`.
   - Tests at [envs/baseline/tests/test_local_hf_client.py](../envs/baseline/tests/test_local_hf_client.py) — mocks `transformers.AutoModelForCausalLM` / `AutoTokenizer` / `BitsAndBytesConfig` + `bitsandbytes` + `torch` via `sys.modules` injection; verifies init (NF4 + bf16), happy-path generation, parse-failure degradation. CI-runnable without a GPU.
3. **Wire alias `qwen` into [envs/eval/scripts/run_eval.py](../envs/eval/scripts/run_eval.py)**:
   - Added `"qwen": "Qwen/Qwen3-8B"` to `MODEL_ALIASES`.
   - Added a `qwen` branch in `_resolve_generators` instantiating `LocalHFClient(model=...)`. One elif; no other runner changes (the Protocol does the rest).
   - **Run with the `baseline` service**, not the `eval` service — eval scripts physically live under `envs/eval/scripts/` but execute against `envs/baseline/`'s venv so heavy torch/bnb deps stay localised. Both envs install `daccord` via path source, so the import surface is identical.
4. **Execute baseline run** (from the repo root, after `uv sync` on the new env):
   ```
   docker compose run --rm baseline uv run python ../eval/scripts/run_eval.py \
     --gold-path ../../data/gold/toy_v1.jsonl \
     --models qwen,groq,gemini \
     --judge meta-llama/llama-4-maverick-17b-128e-instruct \
     --output-csv ../../eval/baseline_toy.csv \
     --run-name baseline-toy-2026-05-25 \
     --seed 42
   ```
   The judge defaults to Llama 4 Scout via Groq per the README ("Eval judge: Llama 4 Scout via Groq free tier"). `--judge gemini-3.1-flash-lite` is the cross-family alternative when Groq is the generator and a non-Groq judge is needed (M4 scenario); Gemini 3.1 Flash Lite free-tier is 15 RPM / 500 RPD.
   Expected outputs: `eval/baseline_toy.csv` with 80 rows (20 pairs × 4 generators: `qwen` local + `groq` Llama 4 Scout + `qwen3` Qwen 3-32B via Groq + `gemini` Gemini 3.1 Flash Lite); MLflow parent run `baseline-toy-2026-05-25-qwen3-llama4` under experiment `daccord-eval` with `dataset_hash` logged as a param; four nested child runs (one per generator) with `tier1_citation_match_overall`, per-jurisdiction breakdowns, `tier2_judge_mean`, judge-bucket counts. First Qwen3-8B run pulls ~5 GB of NF4 weights into the `daccord-hf-cache` volume (one-time).
5. **Update [eval/README.md](../eval/README.md)** with M0 baseline numbers and a pointer to `eval/baseline_toy.csv`. Flag any baseline that scores >0.7 on Tier-1 — that becomes the credible-delta floor the QLoRA fine-tune (M4) must beat to clear the M4 cut criterion (§4).

#### 3B — Lock parser choice (README finalize)

No code. Documentation closure of the M1 parser-bakeoff artifact.

- Update [README.md](../README.md) Architecture section (around the `data/ingest/` row). Append the numeric outcome to the existing Marker line: *"Marker locked for both EN and TH (Thai bake-off: marker recall=1.0 precision=1.0 reading-order=5.0 vs typhoon reading-order=4.0; see [data/parser_bakeoff/summary.md](data/parser_bakeoff/summary.md))."* The Tech Stack PDF processing entry is already past-tense and numeric.
- Keep the public README free of commercial/competitive language. "Marker locked" is a factual technical statement; that's the entire content of the change.

#### M0 closure checklist

- [x] `data/gold/toy_v1.jsonl` committed (0/20 verified — deferred to next MR per 2A decision)
- [x] `eval/baseline_toy.csv` committed (3A artifact, pinned to current `dataset_hash`) — **80 rows, 4 baselines**: Qwen 3-8B (local NF4) + Llama 4 Scout (Groq) + Qwen 3-32B (Groq) + Gemini 3.1 Flash Lite. Judged by Llama 4 Scout (self-judging-bias note in MLflow run tag when `groq` is in the generator pool).
- [x] MLflow `daccord-eval` experiment has the `baseline-toy-2026-05-25` parent run with 2 child runs (qwen + groq); 3rd child added in the Gemini follow-up MR
- [x] README parser-bakeoff section is past-tense with numeric outcome (3B)
- [x] `eval/tokenizer_audit.md` already committed — closes 2C ✓ done
- [x] `data/parser_bakeoff/summary.md` already committed — closes 2D ✓ done
- [x] Costs config covers all 4 free-tier providers (closes 1C); `local_hf` added to Provider Literal for tier 3A
- [x] **GroqJudge** added to scoring layer — judge is Llama 4 Scout via Groq per README ("Eval judge: Llama 4 Scout via Groq free tier"); the §9 docs had drifted to a Gemini judge mid-2B but Groq is the correct default for M0 onward (1000+ RPD headroom for M4's larger eval set)

#### Verification commands (for 3A execution)

```
docker compose run --rm root uv lock --check
docker compose run --rm root uv run ruff check .
docker compose run --rm root uv run ruff format --check .
docker compose run --rm root uv run pyright
docker compose run --rm root uv run pytest
docker compose run --rm eval uv run pytest
docker compose run --rm baseline uv lock --check
docker compose run --rm baseline uv run pytest
```

Plus the baseline command above (live run, hits Groq + Gemini free tiers + loads Qwen3-8B locally).

### 9.4 — Done this MR (2026-05-25)

Snapshot of what closed in this MR so a returning contributor sees the boundary clearly. Each item links to the file or commit-class.

**Code:**
- ✓ `local_hf` added to `Provider` Literal — [src/daccord/costs/config.py](../src/daccord/costs/config.py). Follows the existing `retrieval` precedent (no cap entry, bypasses cost tracker).
- ✓ `LocalHFClient` — 4-bit NF4 Qwen3-8B via `transformers` + `bitsandbytes`; lazy SDK imports so the root env stays slim. [src/daccord/eval/clients.py](../src/daccord/eval/clients.py).
- ✓ `GroqJudge` — Llama 4 Scout via Groq; this is now the default judge per README. [src/daccord/eval/scoring.py](../src/daccord/eval/scoring.py).
- ✓ Shared 10-RPM throttle + Gemini transient-error retry + Groq APIError safety net — [src/daccord/eval/_rpm.py](../src/daccord/eval/_rpm.py) (new). All API clients (Groq + Gemini gen + both judges) call `api_throttle()` after `preflight` and before the SDK call. Local clients (Retrieval, LocalHF) bypass. Groq `BadRequestError`/`RateLimitError` are caught per-call so one bad request doesn't kill the run.
- ✓ Runner refactored to **pair-major iteration** — [src/daccord/eval/runner.py](../src/daccord/eval/runner.py). For each gold pair, every generator is called in sequence (then judged) before moving to the next pair, so per-provider RPM ceilings see lower peak density than the old generator-major batches. CSV row order stays generator-major (wire contract preserved).
- ✓ `qwen` / `qwen3` aliases + `--qwen-model` / `--qwen3-model` flags — [envs/eval/scripts/run_eval.py](../envs/eval/scripts/run_eval.py). Default judge: Llama 4 Scout via Groq (`meta-llama/llama-4-scout-17b-16e-instruct`).
- ✓ Qwen 3 thinking-mode fixes — `enable_thinking=False` in `LocalHFClient.apply_chat_template` (so the local Qwen3-8B response starts at the JSON brace, not `<think>`); `max_tokens` bumped 400 → 2000 in both `GroqClient.generate` and `GroqJudge.judge` so Qwen 3-32B has budget to finish thinking + the JSON answer.
- ✓ Tokenizer audit re-run on Qwen3-8B (the new base) — PASS for th/fr/de/en. [eval/tokenizer_audit.md](../eval/tokenizer_audit.md) regenerated; the audit script path-resolves manifest entries from `framework`/`jurisdiction`/`filename` instead of the host-local `local_path` (fixes container-vs-host path mismatch).
- ✓ `envs/baseline/` sub-project — Python 3.14 + GPU, with torch + bitsandbytes + accelerate + groq + google-genai pinned. Includes mocked LocalHFClient tests (4/4 pass without a GPU).
- ✓ `baseline` compose service — 6th service in [docker-compose.yml](../docker-compose.yml), Dockerfile.cuda image, GPU passthrough, `env_file: .env`.
- ✓ `envs/audit/scripts/extract_pdf_text.py` — formerly untracked PDF helper from 2A drafting; committed.

**Docs:**
- ✓ §9 of this file fully rewritten — all `conda run -n d-accord ...` invocations swapped to `docker compose run --rm <service> ...`; 2A banner records the verification-deferral decision; §9.3 reflects the new `envs/baseline/` + `baseline` compose service.
- ✓ [README.md](../README.md) — "Streamlit (chatbot UI)" leftover replaced with "Streamlit side-by-side comparison + CSV export UI"; SageMaker entry now mentions the HybridRouter inference handler; parser-bakeoff line gets the numeric outcome (3B).
- ✓ [envs/eval/scripts/draft_toy_gold.py](../envs/eval/scripts/draft_toy_gold.py) — docstring example command swapped to docker.
- ✓ Internal dev-environment notes (gitignored): services table updated to 6 entries including `baseline`; GPU-access section covers both bakeoff and baseline.

**Artifacts:**
- ✓ [eval/baseline_toy.csv](../eval/baseline_toy.csv) — 80 rows × 14-column schema (M0 contract). Four baselines, pinned to the current verified-partial gold hash:
  - **Qwen 3-8B** (local, NF4, the locked QLoRA base): Tier-1 = 0.100 (2/20); Tier-2 judge mean = 0.738; 90% ≥ 0.7.
  - **Llama 4 Scout** (Groq, 17B × 16E MoE): Tier-1 = 0.250 (5/20); Tier-2 judge mean = 0.797; 100% ≥ 0.7. *Note: also the judge — Tier-2 self-judging bias caveat applies to this row.*
  - **Qwen 3-32B** (Groq): Tier-1 = 0.200 (4/20); Tier-2 judge mean = 0.792; 100% ≥ 0.7.
  - **Gemini 3.1 Flash Lite** (Google AI Studio): Tier-1 = 0.650 (13/20); Tier-2 judge mean = 0.800; 100% ≥ 0.7.
- ✓ MLflow `daccord-eval` parent run `baseline-toy-2026-05-25` with 2 child runs (one per model) — `dataset_hash` logged as a param via `log_standard_params`.

**Verification (all green this MR):**
- `docker compose run --rm root uv run pytest` — 80 passed
- `docker compose run --rm eval uv run pytest` — 78 passed
- `docker compose run --rm baseline uv run pytest` — 4 passed (mocked LocalHFClient)
- `docker compose run --rm root uv run pyright` — 0 errors, 0 warnings
- `docker compose run --rm root uv run ruff check .` — clean
- `docker compose run --rm root uv run ruff format --check .` — clean

### 9.5 — Deferred to the next MR (pre-tier-4)

Three concrete deferrals. All three must close before tier 4 (corpus parse to markdown) can start.

1. **2A — semantic verification of the 11 claude-extract-only rows.**
   - State after the 2026-05-25 session: 9/20 author-verified (toy_002, 004, 013, 014, 015, 016, 017, 018, 020 — all originally-STUB rows resolved); 11/20 still claude-extract-only (toy_001, 003 (repointed), 005, 006, 007, 008, 009, 010, 011, 012, 019). Citation_ids are confirmed for all 11 (claude extracted them directly from the PDFs); what remains is author-confirming each `*_mechanism` paraphrase is faithful to the regulator's actual wording (no LLM drift).
   - Plan: open each cited PDF page, read the section, compare to the paraphrase in the per-row block of [data/gold/toy_v1_provenance.md](../data/gold/toy_v1_provenance.md). Faster than the STUB pass — citation_ids don't need correction, just paraphrase semantic confirmation.
   - Output: `verified_by` populated for all 20 rows; PENDING notes cleared from JSONL; new `dataset_hash` (changes whenever a paraphrase is touched).
   - Owner: author.
   - Acceptance: `docker compose run --rm eval uv run python scripts/run_eval.py --dry-run --gold-path ../../data/gold/toy_v1.jsonl --verbose` reports new stable hash; STUB-count summary in provenance.md shows "Human-verified: 20".

2. **2A coverage gap — FR Loi I+L native-validation-moat at 2 pairs (below ≥3 target).**
   - State after the 2026-05-25 session: toy_003 and toy_016 were both repointed away from FR Loi I+L during STUB verification, dropping FR coverage from 4 → 2 (toy_004 target, toy_020 source). Below the [data/gold/toy_v1_coverage.md](../data/gold/toy_v1_coverage.md) target of ≥3.
   - Options for next MR:
     - **(a) Accept FR=2** — document the gap in the coverage doc + dev plan; concede that the native-French validation moat is exercised by 2 pairs at M0. M4's larger gold set (≥500 pairs) absorbs this trivially, so the M0 weakening is a localised cost.
     - **(b) Repoint a claude-extract-only row to add FR coverage** — candidates: toy_009 (EU GDPR Art 6 → PDPA-MY Sec 6 consent) → Loi I+L consent provision; or toy_010 (EU GDPR Art 15 → PDPA-MY Sec 30 DSAR) → Loi I+L Art 49 DSAR (already verified for toy_020). Easiest swap: toy_010 → Art 49, since the FR paraphrase exists in toy_020 and can be re-used.
   - Owner: author + claude (one-message proposal/swap exchange like the toy_003 + toy_016 repoints).

3. **§4 M0 strict bar (verified gold)** — closes when 1 and 2 above land. The 3rd-baseline deferral that previously lived here is **resolved** in this MR — `eval/baseline_toy.csv` ships all four comparators (Qwen 3-8B local + Llama 4 Scout + Qwen 3-32B + Gemini 3.1 Flash Lite) against the current verified-partial gold hash. Until the semantic-pass and FR-coverage tasks close, M0 is "artifact bar met, strict bar partial" per the §9.3 closure checklist above. To refresh a single generator after a future verification or model bump, use `--models <alias>` against the new `dataset_hash` — the runner overwrites the CSV per-run, so confirm the full set of `--models` you want before kicking off.

**Gate revision (2026-05-26):** The "must close before tier 4" framing above is **overstated**. The three §9.5 items only affect `eval/baseline_toy.csv` (20-pair toy gold). Tier 4 (corpus parse to markdown) and tier 5 (citation-registry extraction) consume `data/raw/` + `data/raw_manifest.json` and have **zero input from the toy gold**. The §9.5 deferrals are therefore parallel work; a future paraphrase pass only requires re-running the 5-minute baseline eval (`docker compose run --rm baseline uv run python ../eval/scripts/run_eval.py ...`), not re-parsing the corpus. Tier 4 proceeds in this MR.

### 9.6 — Tier 4 (full-corpus PDF→markdown) — **DONE 2026-05-26**

**Status: ✓ Closed.** 13/13 PDFs parsed; R8 spot-check PASS (1.29× regulator-baseline citation density on browser-print docs); manifest at [data/ingest/manifest.jsonl](../data/ingest/manifest.jsonl); R8 report at [data/ingest/r8_spotcheck.txt](../data/ingest/r8_spotcheck.txt). Plan file: `~/.claude/plans/study-tier-4-in-structured-dragonfly.md`.

**What landed (code):**

- ✓ `envs/ingest/` sub-project — Python 3.13 (marker-pdf `pillow<11` ceiling), GPU, depends on `daccord` via path source. Pins: `marker-pdf>=1.10,<2`, `pymupdf>=1.27,<2`, `accelerate>=1.0,<2`. [envs/ingest/pyproject.toml](../envs/ingest/pyproject.toml).
- ✓ `ingest` compose service — 7th service in [docker-compose.yml](../docker-compose.yml). Dockerfile.cuda image (reused from bakeoff/baseline), GPU passthrough, own `daccord-venv-ingest` named volume, `working_dir: /workspace/envs/ingest`.
- ✓ `src/daccord/ingest/` package — permanent home for the tier-4 production parse path, separate from `daccord.bakeoff` which stays the 2D bake-off artifact:
  - [src/daccord/ingest/marker_runner.py](../src/daccord/ingest/marker_runner.py) — `make_converter()` builds the heavy Marker model dict once; `parse_document(pdf_path, out_md_path, converter)` reuses it across 13 documents. `DocumentOutput` (`ValidatedModel`) carries char/page counts + wall-time.
  - [src/daccord/ingest/manifest.py](../src/daccord/ingest/manifest.py) — `IngestManifestEntry` schema (framework, jurisdiction, pdf_relpath, md_relpath, page_count, char_count, marker_version, parsed_at, seconds_elapsed, sha256_pdf, sha256_md, failed, error). `read_manifest` / `write_manifest` JSONL helpers with atomic-replace + sort-by-key.
- ✓ [envs/ingest/scripts/parse_corpus.py](../envs/ingest/scripts/parse_corpus.py) — tier-4 CLI. Flags: `--subset {toy,full}`, `--frameworks <ids>`, `--no-skip-existing`, `--no-mlflow`, `--verbose`. Idempotent: per-doc skip when `md` exists AND prior-run sha256 matches; failed prior runs are retried. Per-doc failure (Marker exception) recorded as `failed=true` in the manifest — sweep continues. Persists manifest after every doc so a mid-run crash loses at most one doc's progress.
- ✓ `daccord-datalab-cache` named volume + `hf_transfer` dep + `HF_HUB_ENABLE_HF_TRANSFER=1` env. Surya (Marker's layout/OCR backend) writes ~3 GB of weights to `/root/.cache/datalab/` on first run; without a named volume those re-download per container exit. Observed download rate over single-stream Docker Desktop networking: **~200 KB/s** (4-hour first run). With `hf_transfer` (Rust-backed parallel chunks): **~13 MB/s** (~65× speedup). The volume is shared with `bakeoff` so tier-2D + tier-4 amortise the download. Pre-existing tracking.py syntax bug (`except A, B, C:` — Py2 syntax, blocks Py3.13 import) fixed in the same commit to unblock the ingest env import path.
- ✓ Tests (all mocked, CI-runnable without a GPU):
  - [tests/test_ingest_manifest.py](../tests/test_ingest_manifest.py) — 5 tests covering read/write roundtrip, sort determinism, upsert-by-key, failed-row null-field serialisation.
  - [envs/ingest/tests/test_marker_runner.py](../envs/ingest/tests/test_marker_runner.py) — 5 tests via `sys.modules` injection of fake `marker.*` + `pymupdf`. Covers `parser_version`, `make_converter`, `parse_document` (writes md + reports stats), converter-reuse-across-docs invariant.
  - [envs/ingest/tests/test_parse_corpus.py](../envs/ingest/tests/test_parse_corpus.py) — 11 tests covering `select_entries` (toy/full/framework allowlist), `output_paths`, `should_skip` (cached-hit + skip-disabled + md-missing + sha-changed + prior-failed branches), `parse_one` failure isolation + happy path with monkeypatched `parse_document`.

**Docs:**

- ✓ [CLAUDE.md](../CLAUDE.md) services table extended to 7 entries; per-env working-dir + GPU-access + "why Python 3.13" sections updated to mention `ingest`; verification block adds `docker compose run --rm ingest uv run pytest`.

**Runs (executed 2026-05-26):**

- ✓ Toy run — `docker compose run --rm ingest uv run python scripts/parse_corpus.py --subset toy --verbose`. 3 EN regulator PDFs in **7 min total**: GDPR (78p, 197K chars, 34.8s), BDSG EN (43p, 145K chars, 20.2s), PDPA-SG (124p, 204K chars, 56.2s). First run also paid the surya-weights download (~3 GB at 13 MB/s = ~5 min) which now lives in the `daccord-datalab-cache` named volume + is reused by all future runs.
- ✓ Full run — `docker compose run --rm ingest uv run python scripts/parse_corpus.py --subset full --verbose`. 10 new parses + 3 cached (`bdsg_en`, `gdpr`, `pdpa_sg` from toy run skipped via idempotent sha256-match): total **~37 min wall-time** for 13 docs. Per-doc table:

  | Framework / file | Pages | Chars | Seconds | Source |
  |---|---:|---:|---:|---|
  | bdsg / bdsg_de_current.pdf | 45 | 176,619 | 22.2 | regulator |
  | bdsg / bdsg_en_current.pdf (cached) | 43 | 145,736 | — | regulator |
  | dpa_2012_ph / dpa_2012_ph.pdf | 26 | 49,977 | 9.8 | regulator |
  | dpa_2012_ph / dpa_2012_ph_irr_amended.pdf | 29 | 104,095 | 13.6 | regulator |
  | dpa_2018 / dpa_2018_current.pdf | 520 | 1,955,194 | 477.2 | **browser-print** |
  | gdpr / reg_2016_679_consolidated.pdf (cached) | 78 | 197,789 | — | regulator |
  | loi_il / loi_78_17_consolidated.pdf | 146 | 268,055 | 856.6 | **browser-print** |
  | pdpa_my / pdpa_my_act709_bilingual.pdf | 191 | 313,206 | 71.5 | regulator |
  | pdpa_my / pdpa_my_amendment_act_a1727_2024.pdf | 10 | 10,152 | 3.1 | regulator |
  | pdpa_sg / pdpa_sg_current.pdf (cached) | 124 | 204,679 | — | regulator |
  | pdpa_th / pdpa_th_english_2019.pdf | 34 | 102,329 | 18.4 | regulator |
  | pdpa_th / pdpa_th_thai_2019.pdf | 44 | 80,512 | 271.8 | regulator |
  | uk_gdpr / uk_gdpr_current.pdf | 148 | 688,164 | 453.1 | **browser-print** |

  Zero failures (`10 parsed (10 ok, 0 failed)`). Big docs (UK DPA 2018, FR Loi I+L, UK-GDPR) trigger Marker's full OCR path because the text layer in browser-print PDFs is non-extractable; per-doc time scales linearly with page count.

- ✓ R8 spot-check — output saved to [data/ingest/r8_spotcheck.txt](../data/ingest/r8_spotcheck.txt). Per-source-page citation-mark density (regex over `§ N | Article N | Art. N | Section N | Sec. N | มาตรา N`):
  - **Regulator avg**: chars/p=2,450.7, cites/p=3.98 (n=10)
  - **Browser-print avg**: chars/p=3,415.2, cites/p=5.14 (n=3)
  - **Ratio**: browser-print / regulator = **1.29× → PASS** (criterion was ≥0.5)
  - Per-doc browser-print: UK-GDPR 6.20 cites/p (148p), UK DPA 2018 3.71 cites/p (520p), FR Loi I+L 5.51 cites/p (146p) — all comfortably above the regulator average of 3.98.
  - R8 fallback (Legifrance API / print-CSS suppression) **not needed**; the surya-Marker stack handles browser-print chrome cleanly.
  - Bilingual PDPA-MY (Act 709) is the regulator-side outlier at 0.26 cites/p — the EN regex doesn't match Malay's "Seksyen" / "Bahagian". Tier 5 will need MY-specific regex; that's a tier-5 concern, not a tier-4 quality issue.

**Artifacts:**

- ✓ `data/ingest/<jur>/<framework>/*.md` — 13 markdown files mirroring `data/raw/` layout. **Committed** (~4 MB total) so tier-5+ work (registry extraction, ensemble) doesn't require a GPU on every fresh clone. Re-parsing only needed on marker-pdf version bumps.
- ✓ [data/ingest/manifest.jsonl](../data/ingest/manifest.jsonl) — 13 rows; `sha256_pdf` for cache invalidation, `sha256_md` for downstream-immutability verification at tier 5; `marker_version` resolved via `importlib.metadata.version("marker-pdf")` since Marker doesn't export `__version__` directly.
- ✓ [data/ingest/r8_spotcheck.txt](../data/ingest/r8_spotcheck.txt) — R8 verdict table (regenerable via `docker compose run --rm ingest uv run python scripts/r8_spotcheck.py`).
- ✓ MLflow experiment `daccord-ingest` — one parent run per `--subset` invocation with per-doc `ingest_seconds__<framework>__<filename>` + `ingest_chars__<framework>__<filename>` metrics; summary `docs_succeeded` / `docs_failed` / `docs_parsed`.

**Verification (all green this MR):**

```
docker compose run --rm root uv lock --check            # in-sync
docker compose run --rm root uv run ruff check .        # clean
docker compose run --rm root uv run ruff format --check .  # clean (pre-existing 3 files unmodified)
docker compose run --rm root uv run pyright             # 0 errors, 0 warnings
docker compose run --rm root uv run pytest              # 85 passed (incl. 5 new ingest manifest tests)
docker compose run --rm ingest uv lock --check          # in-sync
docker compose run --rm ingest uv run pytest            # 16 passed (5 marker_runner + 11 parse_corpus)
docker compose run --rm ingest uv run python scripts/parse_corpus.py --subset toy --verbose
docker compose run --rm ingest uv run python scripts/parse_corpus.py --subset full --verbose
docker compose run --rm ingest uv run python scripts/r8_spotcheck.py
```

### 9.7 — Tier 5 (citation-registry extraction) — **DONE 2026-05-26**

**Status: ✓ Closed.** 9/9 framework registries extracted from the tier-4 parsed markdown; 100% toy-gold base-section recall on every framework present in [data/gold/toy_v1.jsonl](../data/gold/toy_v1.jsonl); R8 follow-up (PDPA-MY Malay regex) resolved. **M1 ("corpus + registries frozen") closed.** Plan file: `~/.claude/plans/plan-tier-5-implementation-stateful-lagoon.md`.

**Decisions resolved up-front (deltas from the original placeholder):**

- **Granularity**: section/article-level with letter-suffixes (e.g. `Article 6` for GDPR; `Section 26A`, `15A`, `22A`, `26D` for PDPA-SG amendments). Subsection precision (`Article 6(1)(a)`) is **not** enumerated — downstream Tier 6A validates predicted citations via prefix-match against the base section. Rationale: Marker emits article numbers as headings, but subsections live in bullet-list structure that varies per framework; enumerating them would 5–10× the work without proportionate downstream value. Tier 1 eval-scoring already operates on the same base-section key space via M0-locked `normalize_citation_id`.
- **Canonical form**: every extracted ID round-trips through [`daccord.eval.scoring.normalize_citation_id`](../src/daccord/eval/scoring.py) (prefix-stripped, lowercased, subsection parens canonicalized) then letter-suffix re-uppercased. `citation_ids` carries the bare canonical form (`"1"`, `"26D"`, `"38"`); the parallel `display_ids` list carries the normalised English heading form (`"Article 1"`, `"Section 26D"`, `"Section 38"`). Registry hits and eval Tier-1 hits **share key space** — non-negotiable, otherwise the registry can't gate ensemble output at Tier 6A.
- **Bilingual frameworks**: one file per `framework_id`, languages merged. `bdsg.json` unions DE `§ N` + EN `Section N` into canonical numeric IDs. `pdpa_my.json` unions Malay `Seksyen N` + EN `Section N`. `pdpa_th.json` unions Thai `มาตรา N` (Thai or Arabic digits) + EN `Section N`. Display form is normalised to English "Section N" for downstream uniformity; the per-language original (`§`, `Seksyen`, `มาตรา`) is not preserved (registry is a constraint set, not a UI artifact).
- **Framework count**: 9 (not the "~5" placeholder estimate). UK has two distinct legal frameworks (UK-GDPR + DPA 2018); README's "8 framework families" counts jurisdictions. 9 matches `data/sources.yaml`'s `framework` keys.
- **M1 anchor**: existing `data/gold/toy_v1.jsonl` (already hand-verified) is the empirical recall test — no separate mini-gold artifact authored. M2's 500-pair gold will further stress-test the registry; missing IDs there get patched incrementally rather than perfectionism here.
- **No timestamps in payload**: `FrameworkRegistry` and `RegistryManifestEntry` deliberately omit `extracted_at` — reruns are byte-identical (verified). `extractor_version` + `source_sha256` carry audit info; git mtime carries the rest. Without this, every CI build would dirty 11 files.

**What landed (code):**

- ✓ `src/daccord/registry/` package — runs in `root` service (no new env):
  - [src/daccord/registry/schema.py](../src/daccord/registry/schema.py) — `FrameworkRegistry` + `RegistryManifestEntry` (`ValidatedModel`); atomic JSON/JSONL I/O (`write_registry`, `read_registry`, `read_manifest`, `write_manifest`, `upsert`).
  - [src/daccord/registry/patterns.py](../src/daccord/registry/patterns.py) — per-framework regex authoring + dispatch table. Reuses [`daccord.bakeoff.scoring.normalize_thai_numerals`](../src/daccord/bakeoff/scoring.py) for `มาตรา` digit normalization. Handles: `\bArticles?\s+(\d+)` (singular + plural; range expansion for "Articles 15 to 22"), `\bSection\s+(\d+[A-Za-z]?)`, `^-\s+\*{0,2}(\d+[A-Za-z]*)\.\s+\S` (PDPA-SG bullet headings), `\*\*(\d+[A-Za-z]*)\.\*\*` (bold inline section headings), `§\s*(\d+[a-z]?)`, `\bSeksyen\s+(\d+[A-Za-z]?)`, `มาตรา\s*([๐-๙0-9]+[A-Za-z]?)`, `\bSEC\.\s*(\d+[A-Za-z]?)`, `^#{1,6}\s+\*{1,2}(\d+[A-Za-z]?)\s+` (UK DPA 2018 bare-number headings), `[Ss]\.\s+(\d+[A-Za-z]?)` (DPA 2018 cross-refs).
  - [src/daccord/registry/extract.py](../src/daccord/registry/extract.py) — `extract_framework()` orchestration: reads multi-doc markdown, dispatches, dedupes, sorts numerically. `compute_toy_gold_recall()` strips subsection suffix and checks base-section membership.
- ✓ [scripts/extract_registry.py](../scripts/extract_registry.py) — tier-5 CLI. Flags mirror [parse_corpus.py](../envs/ingest/scripts/parse_corpus.py): `--ingest-manifest`, `--ingest-root`, `--registry-dir`, `--toy-gold`, `--frameworks <ids>`, `--no-skip-existing`, `--no-mlflow`, `--verbose`. Idempotent: per-framework skip when the prior manifest row's `source_sha256` set matches the current ingest manifest's `sha256_md` set. **Exits non-zero if any framework's toy-gold recall < 1.0** — CI / human reviewer use the exit code as the M1 gate signal.
- ✓ [tests/test_registry.py](../tests/test_registry.py) — 29 tests covering per-framework extractors against representative fixtures (GDPR-style headings, PDPA-SG bullet form, Thai-numeral normalization, BDSG DE+EN merge, etc.), toy-gold recall (strip + match + missing-file edge case), dispatch-table integrity (9 frameworks), atomic-write idempotency, manifest sort + upsert semantics.

**Artifacts (all committed):**

- ✓ `data/registry/{framework_id}.json` × 9 — 9 framework registries (~30 KB total). `.gitignore` updated to allow `/data/registry/` so contributors without a GPU can use them.
- ✓ [data/registry/manifest.jsonl](../data/registry/manifest.jsonl) — 9 rows; per-framework `citation_count`, `cites_per_page`, `toy_gold_recall`, `toy_gold_missing`, `sha256_registry`, `source_sha256` for downstream cache invalidation.
- ✓ [data/registry/summary.md](../data/registry/summary.md) — human-readable table.

**Per-framework results:**

| Framework | Jurisdiction | Citations | Cites/Page | Toy-Gold Recall | Floor (≥) |
|---|---|---:|---:|---:|---:|
| `bdsg` | de | 97 | 1.10 | 1.00 | 70 ✓ |
| `dpa_2012_ph` | ph | 72 | 1.31 | 1.00 | 30 ✓ |
| `dpa_2018` | uk | 288 | 0.55 | 1.00 | 150 ✓ |
| `gdpr` | eu | 92 | 1.18 | 1.00 | 90 ✓ |
| `loi_il` | fr | 126 | 0.86 | 1.00 | 50 ✓ |
| `pdpa_my` | my | 146 | 0.73 | 1.00 | 100 ✓ |
| `pdpa_sg` | sg | 91 | 0.73 | 1.00 | 50 ✓ |
| `pdpa_th` | th | 96 | 1.23 | 1.00 | 70 ✓ |
| `uk_gdpr` | uk | 103 | 0.70 | 1.00 | 90 ✓ |

**What surprised:**

- **GDPR Article 15 has no heading in Marker output.** Between Article 13 (line 232) and Article 16 (post `# S e c t i o n 3` letter-spaced heading), the Article 14 and 15 heading text was dropped. Recall test caught this — fix was to also match plural cross-refs (`\bArticles?\s+(\d+)`) and expand "Articles 15 to 22" ranges. Dense body-text cross-referencing backfills the gap. **General lesson**: M1 gate's empirical anchor (toy-gold recall) is doing real work — first naive run had `gdpr: recall=0.80 missing=['15']`.
- **PDPA-SG section headings are bullet-form, not `Section N`.** The actual headings read `- 13. Consent required` in the TOC and `**13.**` or `**26D.**—(1)` in body — `Section N` only appears in cross-references. First naive run had `pdpa_sg: count=10 cites/p=0.08 recall=0.25`. After adding bullet-form + bold inline patterns, count jumped to 91 and recall to 1.00.
- **R8 follow-up (PDPA-MY Malay regex) confirmed working.** Tier-5 PDPA-MY extracted 146 unique section IDs (vs ~143 estimated structure); the `Seksyen N` regex catches what the tier-4 R8 spot-check's EN-only regex missed.
- **Idempotency required removing all timestamps from the payload.** Initial design carried `extracted_at: datetime` on both `FrameworkRegistry` and `RegistryManifestEntry`; reruns diff'd by 11 files. Removing those fields (audit info now lives in `extractor_version` + `source_sha256` + git mtime) yields byte-identical reruns. Verified by hashing all 11 output files across two consecutive runs.

**Verification (all green this MR):**

```
docker compose run --rm root uv lock --check            # in-sync
docker compose run --rm root uv run ruff check .        # clean
docker compose run --rm root uv run ruff format --check .  # clean (79 files)
docker compose run --rm root uv run pyright             # 0 errors, 0 warnings
docker compose run --rm root uv run pytest              # 114 passed (incl. 29 new registry tests)
docker compose run --rm root uv run python scripts/extract_registry.py
# → "[done] processed=9 skipped=0" + every framework's recall=1.00 → exit 0
docker compose run --rm root uv run python scripts/extract_registry.py
# → "[done] processed=0 skipped=9" (idempotency confirmed)
```

---

## Critical files

- Internal architecture plan (gitignored) — authoritative architecture (Pillar B SageMaker hosting)
- [README.md](../README.md) — parser-bakeoff rationale + eval results updates land here after M4
- `eval/run_eval.py` (to be created — the M0 eval harness is the project's first hard gate)
- `training/train.py` (to be created — HF `transformers` + `peft` + `bitsandbytes` + `trl` stack; small-sweep first at M3)
- `scripts/teardown_endpoint.py` (to be created — committed before Phase 2 first stand-up)
- `scripts/deploy_endpoint.py` (to be created — Phase 2 re-stand-up on demand)
