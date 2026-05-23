# D'accord — Development Sequencing Plan

## Context

D'accord ([README.md](../README.md)) is a private QLoRA fine-tune of Qwen2.5-7B-Instruct for cross-jurisdiction privacy regulation mapping (SEA-4 + EU-spine + UK + DE + FR — 8 framework families). The architecture is already decided in [aws_credential_plan.md](../aws_credential_plan.md) Pillar B. **Architecture is not in scope for this document — sequencing, parallelism, gates, risks, and tooling are.**

### Scope this cycle (Pillar B)

- **No Caravan integration.** Deploy via boto3-direct to SageMaker.
- **Pillar A (Caravan emitter)** deferred. **Pillar C (AgentCore-orchestrated compliance research agent on top of d-accord)** is a future phase, not in this scope but not foreclosed — d-accord's deployed endpoint is the tool Pillar C will consume.
- `code-rag` already credentials RAG / embeddings / semantic search; `invoice-parse` already credentials multi-modal FM in production. d-accord does not redo those.

### JD targets d-accord closes

- **LoRA/QLoRA fine-tuning** (preferred qual)
- **MLOps** — MLflow tracking, model registry, experiment artifacts
- **Regulated industry** — data privacy cross-jurisdiction is the regulated narrative
- **SageMaker hosting** — host the QLoRA adapter on a real-time endpoint
- **Cost optimization & operational excellence** — teardown scripts, cost ceilings, IAM least-privilege

### JD targets *conceded* (per [aws_credential_plan.md](../aws_credential_plan.md))

- Customer-facing thought leadership · FM evaluation as artifact · Secure private-network AI (VPC/PrivateLink/KMS).
- d-accord still produces an eval CSV with per-jurisdiction breakdown — that's good engineering, not the headline credential.

### Landscape context (for scoping awareness, not positioning)

- **OneTrust DataGuidance** — 300+ jurisdiction regulatory research SaaS with AI Copilot (RAG-over-frontier). Adjacent, not competitor; d-accord is a methodology + model artifact, not a SaaS.
- **Harvey AI** — broad legal AI ($11B val); frontier-model approach; citation hallucination remains a known weakness on niche SEA regs.
- **SaulLM-7B** — closest methodological cousin (7B legal specialist, pretrained on 19M docs). Different task; useful prior-art reference.

**Effort baseline**: ~3 wk (~0.5–0.7 EM). AWS cost ceiling **$50–100**.

### Phased execution

- **Phase 1 — Local validation (M0–M4)**: full data pipeline + QLoRA training on RTX 5080 + three-tier eval. **All credentials except SageMaker hosting close here.** No AWS-runtime dependency, no AWS spend risk.
- **Phase 2 — SageMaker hosting (M5)**: deploy adapter to endpoint via boto3, smoke test, capture, tear down. Decoupled from Phase 1; can be triggered when an interview window or other concrete reason justifies the $50–100 spend.
- **Phase C — AgentCore agent (future, separate repo)**: consumes d-accord's deployed endpoint as a tool. Out of scope here.

---

## 1. Tooling Decision — Training Framework

| Concern | Recommendation |
|---|---|
| Language | **Python** — no realistic alternative for QLoRA-7B; HF/PEFT/bitsandbytes is the ecosystem |
| Framework | **`transformers` + `peft` + `bitsandbytes` + `trl` + `accelerate`** as the default |
| Speed/VRAM fallback | **Unsloth** if M3 small-sweep OOMs or trains >2× slower than expected. Drop-in over HF + PEFT; fused Triton kernels for LoRA forward/backward + FlashAttention-2 tuned for LoRA → ~2× throughput, ~40–50% lower peak VRAM. Qwen2.5-7B supported. Trade-off: lags on newest models; nf4 4-bit only |
| Avoid | Axolotl / LLaMA-Factory — YAML-config wrappers. Engineering choices (LoRA rank, target modules, LR schedule, collator) live in config not Python, so a portfolio reviewer sees "ran the wrapper" not "made deliberate choices". Fine in a production team; wrong for a credential project |
| Experiment tracking | **MLflow autolog** wired from the toy run forward, not added later |

---

## 2. Task Graph

**Notation**: tasks are numbered by tier. **Numbers are ordinal — tier *N+1* cannot start until tier *N* closes.** **Letters within a tier are parallel — 1A, 1B, 1C run concurrently.** Gates `[M*]` mark milestone checkpoints; see §4 for DoD and cut criteria.

### Phase 1 — Local validation

| Tier | Tasks | Type | Notes |
|---|---|---|---|
| **1** | 1A repo skeleton + lockfile · 1B MLflow + autolog plumbing · 1C per-provider daily caps · 1D start PDF corpus download | parallel | All independent; day-1 |
| **2** | 2A 20-pair hand-built toy gold · 2B eval harness (citation match + judge) · 2C tokenizer audit (Thai/FR/DE) · 2D Thai parser bake-off on 5-page sample (Marker / PaddleOCR+spatial / Typhoon) | parallel | No full corpus needed |
| **3** | 3A baselines on toy gold (base Qwen + Sonnet + GPT-4o) · 3B lock parser choice from 2D results | parallel | 3A needs 2A+2B; 3B needs 2D |
|  | **[M0 gate]** | | tokenizer passes · baselines captured · parser locked |
| **4** | Parse all PDFs to markdown (Marker EN, bake-off winner TH) | sequential | Needs 1D complete + 3B parser choice. **Watch R8**: 3 sources (UK-GDPR, UK DPA 2018, FR Loi I+L) come from browser print-to-PDF (Légifrance/legislation.gov.uk expose no scraper-friendly consolidated PDF) — 5–60× larger than regulator-issued PDFs, layout may confuse Marker |
| **5** | Citation registry extraction per framework | sequential | |
|  | **[M1 gate]** | | corpus + registries frozen |
| **6** | 6A ensemble prompt + JSON schema (citations constrained to registry from 5) · 6B tiering script | parallel | |
| **7** | 7A ensemble generation — Sonnet + GPT-4o + Qwen-72B (~3d **async**) · 7B splits script · 7C hand-validate completed framework-pairs as they land | parallel | 7A is the long async job; 7B/7C fill the wait |
| **8** | Tiering (HIGH/MED/LOW/SALVAGE) + complete hand-validation + HIGH-tier per-jurisdiction spot-check | sequential | Needs 7A complete + all 7C |
| **9** | Gold freeze (≥500 pairs) + jurisdiction-disjoint train/val/test splits + dataset SHA | sequential | |
|  | **[M2 gate]** | | gold + splits frozen with version hash |
| **10** | 10A `training/train.py` (HF `transformers` + `peft` + `bitsandbytes` + `trl`) · 10B small-sweep config | parallel | 10A can actually start during tier 7 idle time |
| **11** | Small-sweep — 200 pairs × 1 epoch | sequential | Validates MLflow plumbing, adapter save/reload, OOM headroom |
|  | **[M3 gate]** | | adapter saves/reloads · MLflow logs run + SHA · no OOM at target seq_len (else swap to Unsloth) |
| **12** | 12A full QLoRA train + small hyperparam sweep (~overnight **async**) · 12B three-tier eval script · 12C draft Phase 2 deploy/teardown scripts | parallel | 12A async; 12B/12C fill the wait |
| **13** | Three-tier eval, per-jurisdiction + per-language breakdown | sequential | |
|  | **[M4 gate]** | | Phase 1 done — eval CSV + MLflow history + adapter on disk |

### Phase 2 — SageMaker hosting (triggered separately)

| Tier | Tasks | Type | Notes |
|---|---|---|---|
| **14** | 14A IAM user `d-accord-dev` + scoped S3 bucket with versioning · 14B AWS Budgets alarm ($50/$100) · 14C **teardown scripts committed before any stand-up** · 14D adapter packaged to SageMaker S3 layout | parallel | Stand-up (tier 15) blocked until 14C is in git |
| **15** | SageMaker endpoint stand-up via boto3 (`ml.g5.xlarge`) | sequential | ~5–10 min cold start |
| **16** | Smoke test 10 prompts via Streamlit chatbot | sequential | |
| **17** | Capture — recording + screenshots | sequential | |
| **18** | Endpoint teardown | sequential | Within 48 h of capture · spend <$100 |
|  | **[M5 gate]** | | recording captured · endpoint down · adapter remains in S3 |

---

## 3. Execution Notes

**The two long async jobs** are **7A** (ensemble generation, ~3 days) and **12A** (full QLoRA train, overnight). 1D (PDF download) is also unattended but short. These are the only places idle time can accumulate — launch each *before* sitting down to its tier's parallel tasks (7B/7C and 12B/12C).

**Ensemble checkpointing**: write `data/ensemble/raw/{framework_pair}__{model}.jsonl` as each batch lands. Resume logic skips completed pairs so a Together rate-limit at hour 4 of 6 burns zero re-spend.

**Rough timeline** (solo dev, baseline velocity):

| Days | Tiers | Closes |
|---|---|---|
| d1–3 | 1 + 2 + 3 | **M0** |
| d4–6 | 4 + 5 | **M1** (end of week 1) |
| d7–10 | 6 + 7 + 8 + 9 | **M2** |
| d11–12 | 10 + 11 | **M3** |
| d13–15 | 12 + 13 | **M4** — Phase 1 done |
| later | 14 → 18 | **M5** (Phase 2, triggered separately) |

---

## 4. Milestone Gates

### Phase 1 — Local validation (M0 → M4)

All ML substance happens here. No AWS resources stood up. Zero AWS spend risk. If Phase 2 is deferred indefinitely, Phase 1 still constitutes a complete portfolio artifact (eval CSV + MLflow runs + local-inference recording + README).

### M0 — Eval Bar Locked (end of d3)

- **DoD**: 20-pair toy gold built · eval harness runs end-to-end · baselines captured (base Qwen + Sonnet + GPT-4o) on toy · tokenizer audit committed
- **Artifact**: `eval/baseline_toy.csv` + `eval/tokenizer_audit.md`
- **Cut criterion**: tokenizer audit shows Qwen2.5-7B fragments Thai at >2 tokens/char average → escalate immediately: swap base (SeaLLM-v3, Typhoon-7B) or descope Thai. **Decide here, not at training time.**

### M1 — Corpus + Registry Frozen (end of week 1)

- **DoD**: All 8 framework families parsed to markdown · registries extracted · parser-choice rationale in README
- **Artifact**: `data/registry/*.json` per framework + parser bake-off score table
- **Cut criterion**: Thai bake-off has no clear winner OR all three fail on Royal Gazette amendments → drop Royal Gazette (keep PDPA-TH core only); document the cut.

### M2 — Gold Set Frozen (~d10)

- **DoD**: ≥500 hand-validated gold pairs · ensemble outputs checkpointed · HIGH-tier stratified spot-check shows no jurisdiction <80% sample quality · jurisdiction-disjoint train/test split committed with dataset hash
- **Artifact**: `data/gold/gold_v1.jsonl` + `data/splits/{train,val,test}.jsonl` + spot-check report
- **Cut criterion**: gold <300 by d10 → drop Malaysia + Philippines (cheap completers); two-native-language story stays intact.

### M3 — Small-Sweep Validated (~d12)

- **DoD**: 1 epoch × 200 pairs trains end-to-end · loss curve sensible · adapter saves+reloads cleanly · MLflow autolog shows the run with adapter SHA logged
- **Artifact**: MLflow screenshot + sanity-check inference output
- **Cut criterion**: OOM at QLoRA-7B on the 5080 → drop max_seq_len 4096→2048, add gradient checkpointing, micro-batch 1 + grad-accum 16. If still OOM, **swap to Unsloth** before full train.

### M4 — Eval Delta Proven (~d15) — Phase 1 done

- **DoD**: Three-tier eval against M2 gold · per-jurisdiction + per-language breakdown · delta vs M0 baselines numerically captured
- **Artifact**: `eval/results_v1.csv` + per-jurisdiction breakdown table; MLflow run history; updated README
- **Cut criterion**: fine-tune delta vs base Qwen <5% on Tier-1 citation accuracy AND negative vs Sonnet on every jurisdiction → **do not push to SageMaker**. Document the honest negative result in the eval CSV. Tear down API spend.

### Phase 2 — SageMaker hosting (M5)

Trigger when M4 has a publishable delta AND there's a concrete reason (interview window, runway) to absorb the AWS spend. Until then, Phase 1 artifacts are sufficient.

### M5 — Endpoint Live, Captured, Torn Down (~2–3 days when triggered)

- **DoD**: Endpoint live · Streamlit chatbot answers 5 test prompts correctly · short screen recording captured · **endpoint torn down** · S3 artifact remains (cheap)
- **Artifact**: Recording + 4–6 screenshots + adapter S3 URI
- **Cut criterion**: endpoint burn rate puts $100 ceiling at risk → tear down within 48 h of capture. **The durable artifact is the recording, not the running endpoint.** Re-stand-up on demand from `scripts/deploy_endpoint.py` (~5–10 min cold start) for interviews.

---

## 5. LLM Fine-Tuning Practices to Layer In

- **Baseline-before-fine-tune (M0)** — non-negotiable. The "specialist beats frontier on under-represented jurisdictions" claim requires numerical proof against base Qwen *and* Sonnet/GPT-4o on the same eval set. No baseline → no defensible claim.
- **Tokenizer audit (M0)** — minutes to run. Qwen2.5's ~150k vocab should handle Thai/FR/DE; verify empirically. Bytefallback >20% on Thai = hard stop.
- **Small-sweep before full train (M3)** — 200 pairs, 1 epoch, ~30 min. Validates adapter save/reload, MLflow autolog capture, loss-curve shape, OOM behavior at full seq_len. Cheaper to discover plumbing breakage on 200 pairs than 5000.
- **MLflow autolog from the toy run** — log every run from day-1, including failed/aborted ones. A full run history is itself an artifact.
- **Per-jurisdiction breakdowns in metrics** — every eval row is `(jurisdiction_source, jurisdiction_target, citation_match, judge_score)`. Aggregate per-jurisdiction-pair in the CSV.
- **Jurisdiction-disjoint test slice** — hold out specific control areas (e.g., breach notification across jurisdictions in test; data subject rights in train). Detects overfitting to specific control families.
- **Reproducibility**: pin `torch`/`transformers`/`peft`/`bitsandbytes`/`trl` versions in a lockfile; set seeds (`torch`, `numpy`, `random`, `transformers.set_seed`); log adapter SHA256 + git commit hash in MLflow params; hash and version the gold dataset; `eval/results_v1.csv` references the hash explicitly.

---

## 6. Cloud / Cost Practices

- **Phase 1 spend** is API only (Sonnet/GPT-4o/Together). One row per day in `costs/daily.csv` committed to repo. Per-provider soft caps before pipeline kickoff: Anthropic $30/d, OpenAI $20/d, Together $15/d.
- **Phase 2 SageMaker discipline**: `ml.g5.xlarge` ≈ $1.40/hr; **target <48 h total live**; stand up → smoke test (10 prompts) → capture → tear down (~2 h live). Re-stand-up on demand from `scripts/deploy_endpoint.py`; budget for ~5–10 min cold start during interviews.
- **IAM least-privilege**: dedicated user `d-accord-dev`, never root; two policies: `s3:* on arn:aws:s3:::d-accord-artifacts/*` and `sagemaker:* on resources tagged Project=d-accord`.
- **S3 versioning** enabled on `d-accord-artifacts` (trivial cost, prevents adapter clobber).
- **Teardown as committed code** before first stand-up (`scripts/teardown_endpoint.py`, `scripts/teardown_all.py --nuke`).
- **API spend resilience**: ensemble outputs checkpointed per `(framework_pair, model)` to `data/ensemble/raw/`; resume logic skips completed pairs.
- **Project tag** `Project=d-accord` on every AWS resource for cost attribution.

---

## 7. Risk Register

| # | Risk | Likelihood | Impact | Mitigation | Early-warning signal |
|---|---|---|---|---|---|
| R1 | Thai parser bake-off has no clear winner; registries unreliable | Medium | High (kills SEA differentiation) | Bake-off in M0/M1 on 5-page sample; cut = drop Royal Gazette, keep core PDPA-TH | Day-2 bake-off scores cluster within 5% across all 3 parsers |
| R2 | Ensemble agreement collapses on SEA frameworks (shared blind spots) | Medium-High | High (gold set thin) | Stratified human spot-check on HIGH-tier *per jurisdiction* (M2); cut = drop weakest 2 SEA jurisdictions | HIGH-tier spot-check <70% on any one jurisdiction |
| R3 | Gold dataset stalls <500 pairs by M2 | Medium | Medium (eval power weakens) | Cut to 6 jurisdictions (drop MY + PH); reuse HIGH-tier with 10% audit as proxy | By d8, validation pace projects <400 pairs by d10 |
| R4 | Qwen2.5-7B tokenizer fragments Thai/FR worse than expected | Low-Medium | High (kills language-validation moat) | M0 tokenizer audit before training plumbing; swap to SeaLLM-v3 or Typhoon-7B if Thai byte-fallback >20% | Audit shows >2.5 tokens/char on Thai or byte-fallback artifacts |
| R5 | RTX 5080 OOM at QLoRA-7B with full seq_len | Medium | Medium (slows training) | M3 small-sweep catches before full train; mitigations: max_seq_len 2048, gradient checkpointing, micro-batch 1 + grad-accum 16; **swap to Unsloth** if needed | Small-sweep OOMs at any seq_len >1024 |
| R6 | Fine-tune delta vs frontier baseline marginal or negative | Medium | High (kills headline credential claim) | M0 baselines set expectation early; M4 cut = document honest negative result and skip Phase 2 | Tier-1 citation accuracy delta vs base Qwen <5% by mid-train checkpoint |
| R7 | API cost overrun (ensemble runs unexpectedly expensive) | Low-Medium | Medium ($ ceiling breach) | Per-provider daily caps; checkpointed ensemble outputs make retries free | Daily spend >$25/d for 2 consecutive days |
| R8 | Browser-print PDFs (UK-GDPR, UK DPA 2018, FR Loi I+L) parse noisily under Marker — Légifrance + legislation.gov.uk expose no scraper-friendly consolidated PDF, so 1D fell back to print-to-PDF (5–60× larger than regulator-issued PDFs, embedded page chrome) | Medium | Medium (registry drift on EU-spine + UK) | At tier 4, spot-check Marker output on these 3 vs a regulator-issued reference (e.g., BDSG); fallback = headless-browser PDF export with print-CSS suppression, or source from Legifrance API / legislation.gov.uk Atom feed | Tier 4 Marker output for UK/FR contains >2× the unrecognised tokens or broken citation IDs vs auto-downloaded sources |

---

## 8. Verification

End-to-end DoD for this development cycle:

**Phase 1**:
1. M0–M4 all hit (or their cut criteria triggered with documented decisions)
2. `eval/results_v1.csv` shipped with per-jurisdiction breakdown
3. MLflow run history populated with adapter SHA + dataset hash linkage
4. README updated with parser-bakeoff rationale and eval results
5. Reproducibility: locked Python env · seeded train script · dataset hash referenced in `eval/results_v1.csv`

**Phase 2 (when triggered)**:
6. Adapter packaged in SageMaker-compatible S3 layout · adapter S3 URI documented
7. Endpoint deploy + teardown scripts committed and tested · `Project=d-accord` tag spend remains <$100
8. Endpoint live for capture session · recording + screenshots in repo · endpoint torn down
9. IAM policy JSON committed; S3 versioning verified

## Critical files

- [aws_credential_plan.md](../aws_credential_plan.md) — authoritative architecture
- [README.md](../README.md) — parser-bakeoff rationale + eval results updates land here after M4
- `c:\Users\paulx\Documents\portfolio\invoice-parse\` — PaddleOCR + spatial-clustering code to reuse for the Thai parser bake-off candidate #2
- `eval/run_eval.py` (to be created — the M0 eval harness is the project's first hard gate)
- `training/train.py` (to be created — HF `transformers` + `peft` + `bitsandbytes` + `trl` stack; small-sweep first at M3)
- `scripts/teardown_endpoint.py` (to be created — committed before Phase 2 first stand-up)
- `scripts/deploy_endpoint.py` (to be created — Phase 2 re-stand-up on demand)
