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
- **Hybrid serving with provenance tagging** — retrieval-first + QLoRA fallback at the endpoint; per-response `gold-retrieval` / `fine-tune-generalization` tagging is an operational maturity signal beyond a vanilla adapter-only endpoint
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
| **1** | 1A repo skeleton + lockfile · 1B MLflow + autolog plumbing · 1C per-provider RPD caps for free-tier APIs (Groq/Cerebras/Google AI Studio/DeepSeek) · 1D start PDF corpus download | parallel | All independent; day-1 |
| **2** | 2A 20-pair hand-built toy gold · 2B eval harness (citation match + judge) · 2C tokenizer audit (Thai/FR/DE) · 2D Thai parser bake-off on 5-page sample (Marker vs Typhoon-OCR — **Marker locked**, see `data/parser_bakeoff/summary.md`) | parallel | No full corpus needed |
| **3** | 3A baselines on toy gold (base Qwen 2.5-7B + Llama 3.x 70B via Groq + Gemini 2.x Flash via Google AI Studio) · 3B lock parser choice from 2D results | parallel | 3A needs 2A+2B; 3B needs 2D; baselines are all free-tier OSS to match the tier 7A ensemble |
|  | **[M0 gate]** | | tokenizer passes · baselines captured · parser locked |
| **4** | Parse all PDFs to markdown (Marker, locked for both EN and TH) | sequential | Needs 1D complete + 3B parser choice. **Watch R8**: 3 sources (UK-GDPR, UK DPA 2018, FR Loi I+L) come from browser print-to-PDF (Légifrance/legislation.gov.uk expose no scraper-friendly consolidated PDF) — 5–60× larger than regulator-issued PDFs, layout may confuse Marker |
| **5** | Citation registry extraction per framework | sequential | |
|  | **[M1 gate]** | | corpus + registries frozen |
| **6** | 6A ensemble prompt + JSON schema (citations constrained to registry from 5) · 6B tiering script | parallel | |
| **7** | 7A ensemble generation — 4-model OSS via free-tier APIs (Llama 3.x 70B/Groq + Qwen 2.5/3 70B/Cerebras or local + Gemini 2.x Flash/Google AI Studio + DeepSeek V3 or Mixtral 8x22B) (~3d **async**) · 7B splits script · 7C hand-validate completed framework-pairs as they land | parallel | 7A async pacing now driven by free-tier RPD limits, not API spend; 7B/7C fill the wait |
| **8** | Tiering (HIGH/MED/LOW/SALVAGE) + complete hand-validation + HIGH-tier per-jurisdiction spot-check | sequential | Needs 7A complete + all 7C |
| **9** | Gold freeze (≥500 pairs) + jurisdiction-disjoint train/val/test splits + dataset SHA | sequential | |
|  | **[M2 gate]** | | gold + splits frozen with version hash |
| **10** | 10A `training/train.py` (HF `transformers` + `peft` + `bitsandbytes` + `trl`) · 10B small-sweep config | parallel | 10A can actually start during tier 7 idle time |
| **11** | Small-sweep — 200 pairs × 1 epoch | sequential | Validates MLflow plumbing, adapter save/reload, OOM headroom |
|  | **[M3 gate]** | | adapter saves/reloads · MLflow logs run + SHA · no OOM at target seq_len (else swap to Unsloth) |
| **12** | 12A full QLoRA train + small hyperparam sweep (~overnight **async**) · 12B three-tier eval script + retrieval baseline (MPNet+FAISS over train-split source clauses) + `build_retrieval_index.py` · 12C draft Phase 2 deploy/teardown scripts + hybrid inference handler (`publish/sagemaker_handler.py`) + Streamlit side-by-side app (`consumer/app.py`) | parallel | 12A async; 12B/12C fill the wait. 12C is now substantive (~2–3 d) — don't rush. |
| **13** | Three-tier eval across 4 comparators (fine-tune + base Qwen + Llama 70B + retrieval) run twice with `--slice-tag in-domain` and `--slice-tag out-of-domain`; per-jurisdiction + per-language breakdown aggregated from CSV rows | sequential | Slice tag goes to MLflow run metadata, not per-row (CSV contract stable per [eval/README.md](../eval/README.md)). |
|  | **[M4 gate]** | | Phase 1 done — eval CSV + MLflow history + adapter on disk |

### Phase 2 — SageMaker hosting (triggered separately)

| Tier | Tasks | Type | Notes |
|---|---|---|---|
| **14** | 14A IAM user `d-accord-dev` + scoped S3 bucket with versioning · 14B AWS Budgets alarm ($50/$100) · 14C **teardown scripts committed before any stand-up** · 14D adapter + retrieval index + embedder snapshot + custom inference handler packaged to SageMaker S3 layout via `publish/package_model.py` | parallel | Stand-up (tier 15) blocked until 14C is in git |
| **15** | SageMaker endpoint stand-up via boto3 (`ml.g5.xlarge`) | sequential | ~5–10 min cold start |
| **16** | Smoke test 10 source clauses via side-by-side comparison view (5 in-domain, 5 out-of-domain); verify provenance tags (`gold-retrieval` / `fine-tune-generalization`) return correctly + CSV export round-trip | sequential | |
| **17** | Capture — recording + screenshots | sequential | |
| **18** | Endpoint teardown | sequential | Within 48 h of capture · spend <$100 |
|  | **[M5 gate]** | | recording captured · endpoint down · adapter remains in S3 |

---

## 3. Execution Notes

**The two long async jobs** are **7A** (ensemble generation, ~3 days) and **12A** (full QLoRA train, overnight). 1D (PDF download) is also unattended but short. These are the only places idle time can accumulate — launch each *before* sitting down to its tier's parallel tasks (7B/7C and 12B/12C). 7A's ~3-day duration is now driven by free-tier RPD pacing across providers (Groq ~14400 RPD, Cerebras free-tier daily quota, Google AI Studio 1500 RPD, DeepSeek), not by API spend or rate limits — the ensemble runner schedules per-provider request streams against current free-tier quotas verified at run time.

**Ensemble checkpointing**: write `data/ensemble/raw/{framework_pair}__{model}.jsonl` as each batch lands. Resume logic skips completed pairs so a free-tier RPD cap hit at hour 4 of 6 burns zero re-work; the runner waits for the daily reset and continues.

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

- **DoD**: 20-pair toy gold built · eval harness runs end-to-end · baselines captured (base Qwen 2.5-7B + Llama 3.x 70B via Groq + Gemini 2.x Flash via Google AI Studio) on toy · tokenizer audit committed
- **Artifact**: `eval/baseline_toy.csv` + `eval/tokenizer_audit.md`
- **Cut criterion**: tokenizer audit shows Qwen2.5-7B fragments Thai at >2 tokens/char average → escalate immediately: swap base (SeaLLM-v3, Typhoon-7B) or descope Thai. **Decide here, not at training time.**

### M1 — Corpus + Registry Frozen (end of week 1)

- **DoD**: All 8 framework families parsed to markdown · registries extracted · parser-choice rationale in README
- **Artifact**: `data/registry/*.json` per framework + parser bake-off score table
- **Cut criterion**: Thai bake-off has no clear winner OR both candidates fail on Royal Gazette amendments → drop Royal Gazette (keep PDPA-TH core only); document the cut. (**Resolved 2D**: Marker won with 48/48 perfect citation extraction on PDPA-TH original; cut not triggered.)

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
- **Cut criterion**: fine-tune delta vs base Qwen <5% on Tier-1 citation accuracy AND negative vs Llama 3.x 70B on every jurisdiction AND **no advantage over the retrieval baseline on the out-of-domain slice** → **do not push to SageMaker**. The retrieval-baseline qualifier is what makes the cut honest: if retrieval ties or beats fine-tune everywhere, ship as retrieval-only (architecture pivot), don't ship the heavier serving stack just to preserve the credential framing. Document the honest negative result in the eval CSV. (No paid-API spend to tear down — ensemble + baselines + judge are all free-tier OSS.)

### Phase 2 — SageMaker hosting (M5)

Trigger when M4 has a publishable delta AND there's a concrete reason (interview window, runway) to absorb the AWS spend. Until then, Phase 1 artifacts are sufficient.

### M5 — Endpoint Live, Captured, Torn Down (~2–3 days when triggered)

- **DoD**: Endpoint live · side-by-side comparison view returns retrieval + fine-tune + base outputs with provenance tags for 5 test source clauses · CSV export verified · short screen recording captured · **endpoint torn down** · S3 artifact remains (cheap)
- **Artifact**: Recording + 4–6 screenshots + adapter S3 URI
- **Cut criterion**: endpoint burn rate puts $100 ceiling at risk → tear down within 48 h of capture. **The durable artifact is the recording, not the running endpoint.** Re-stand-up on demand from `scripts/deploy_endpoint.py` (~5–10 min cold start) for interviews.

---

## 5. LLM Fine-Tuning Practices to Layer In

- **Baseline-before-fine-tune (M0)** — non-negotiable. The "specialist achieves citation-faithful structural mapping with native-language validation moats" claim requires numerical proof against base Qwen 2.5-7B *and* strong OSS comparators (Llama 3.x 70B via Groq, Gemini 2.x Flash via Google AI Studio) on the same eval set. No baseline → no defensible claim.
- **Tokenizer audit (M0)** — minutes to run. Qwen2.5's ~150k vocab should handle Thai/FR/DE; verify empirically. Bytefallback >20% on Thai = hard stop.
- **Small-sweep before full train (M3)** — 200 pairs, 1 epoch, ~30 min. Validates adapter save/reload, MLflow autolog capture, loss-curve shape, OOM behavior at full seq_len. Cheaper to discover plumbing breakage on 200 pairs than 5000.
- **MLflow autolog from the toy run** — log every run from day-1, including failed/aborted ones. A full run history is itself an artifact.
- **Per-jurisdiction breakdowns in metrics** — every eval row is `(jurisdiction_source, jurisdiction_target, citation_match, judge_score)`. Aggregate per-jurisdiction-pair in the CSV.
- **Jurisdiction-disjoint test slice** — hold out specific control areas (e.g., breach notification across jurisdictions in test; data subject rights in train). Detects overfitting to specific control families.
- **Reproducibility**: pin `torch`/`transformers`/`peft`/`bitsandbytes`/`trl` versions in a lockfile; set seeds (`torch`, `numpy`, `random`, `transformers.set_seed`); log adapter SHA256 + git commit hash in MLflow params; hash and version the gold dataset; `eval/results_v1.csv` references the hash explicitly.

---

## 6. Cloud / Cost Practices

- **Phase 1 spend** is effectively $0 — ensemble (7A) + LLM-as-judge (13) both run on open-source models via free-tier APIs (Groq, Cerebras, Google AI Studio, DeepSeek direct). Only paid Phase 1 cost is ~$5–10 LlamaParse fallback if Marker fails on a specific document. One row per day in `costs/daily.csv` committed to repo with **request-count** entries against per-provider RPD caps (Groq ~14400 RPD, Google AI Studio ~1500 RPD; Cerebras + DeepSeek quotas verified at run time). Hard $5/provider paid-spill ceiling if free tier exhausts during a run.
- **Phase 2 SageMaker discipline**: `ml.g5.xlarge` ≈ $1.40/hr; **target <48 h total live**; stand up → smoke test (10 prompts) → capture → tear down (~2 h live). Re-stand-up on demand from `scripts/deploy_endpoint.py`; budget for ~5–10 min cold start during interviews. Cold start now also loads the MPNet embedder + FAISS index alongside the 7B adapter (~1–2 GB additional read; negligible time impact vs the adapter load).
- **IAM least-privilege**: dedicated user `d-accord-dev`, never root; two policies: `s3:* on arn:aws:s3:::d-accord-artifacts/*` and `sagemaker:* on resources tagged Project=d-accord`.
- **S3 versioning** enabled on `d-accord-artifacts` (trivial cost, prevents adapter clobber).
- **Teardown as committed code** before first stand-up (`scripts/teardown_endpoint.py`, `scripts/teardown_all.py --nuke`).
- **API spend resilience**: ensemble outputs checkpointed per `(framework_pair, model)` to `data/ensemble/raw/`; resume logic skips completed pairs.
- **Project tag** `Project=d-accord` on every AWS resource for cost attribution.

---

## 7. Risk Register

| # | Risk | Likelihood | Impact | Mitigation | Early-warning signal |
|---|---|---|---|---|---|
| R1 | Thai parser bake-off has no clear winner; registries unreliable | Medium | High (kills SEA differentiation) | Bake-off in M0/M1 on 5-page sample; cut = drop Royal Gazette, keep core PDPA-TH | ~~Day-2 bake-off scores cluster within 5% across all 3 parsers~~ **Resolved 2D**: Marker locked (48/48 citation extraction, ~2× faster than Typhoon; PaddleOCR candidate dropped pre-execution). |
| R2 | Ensemble agreement collapses on SEA frameworks (shared blind spots) | Medium-High | High (gold set thin) | Stratified human spot-check on HIGH-tier *per jurisdiction* (M2); cut = drop weakest 2 SEA jurisdictions | HIGH-tier spot-check <70% on any one jurisdiction |
| R3 | Gold dataset stalls <500 pairs by M2 | Medium | Medium (eval power weakens) | Cut to 6 jurisdictions (drop MY + PH); reuse HIGH-tier with 10% audit as proxy | By d8, validation pace projects <400 pairs by d10 |
| R4 | Qwen2.5-7B tokenizer fragments Thai/FR worse than expected | Low-Medium | High (kills language-validation moat) | M0 tokenizer audit before training plumbing; swap to SeaLLM-v3 or Typhoon-7B if Thai byte-fallback >20% | Audit shows >2.5 tokens/char on Thai or byte-fallback artifacts |
| R5 | RTX 5080 OOM at QLoRA-7B with full seq_len | Medium | Medium (slows training) | M3 small-sweep catches before full train; mitigations: max_seq_len 2048, gradient checkpointing, micro-batch 1 + grad-accum 16; **swap to Unsloth** if needed | Small-sweep OOMs at any seq_len >1024 |
| R6 | Fine-tune delta vs frontier baseline marginal or negative | Medium | High (kills headline credential claim) | M0 baselines set expectation early; M4 cut = document honest negative result and skip Phase 2 | Tier-1 citation accuracy delta vs base Qwen <5% by mid-train checkpoint |
| R7 | Free-tier RPD exhaustion stalls ensemble (Phase 1 spend is $0 OSS-via-free-tier; risk is rate, not cost) | Low | Low (delays, not $ ceiling breach) | Per-provider RPD caps + checkpointed ensemble outputs; runner waits for daily reset and resumes; hard $5/provider paid-spill ceiling if absolutely needed | Free-tier RPD exhaustion on 2 providers within same eval-pass window |
| R8 | Browser-print PDFs (UK-GDPR, UK DPA 2018, FR Loi I+L) parse noisily under Marker — Légifrance + legislation.gov.uk expose no scraper-friendly consolidated PDF, so 1D fell back to print-to-PDF (5–60× larger than regulator-issued PDFs, embedded page chrome) | Medium | Medium (registry drift on EU-spine + UK) | At tier 4, spot-check Marker output on these 3 vs a regulator-issued reference (e.g., BDSG); fallback = headless-browser PDF export with print-CSS suppression, or source from Legifrance API / legislation.gov.uk Atom feed | Tier 4 Marker output for UK/FR contains >2× the unrecognised tokens or broken citation IDs vs auto-downloaded sources |
| R9 | Retrieval baseline dominates fine-tune on in-domain AND ties on out-of-domain → architectural claim (specialist value-add) weakens | Medium | Medium (reframes the credential, doesn't kill it) | Ship the eval CSV honestly; reframe README from "fine-tune is the engine" to "hybrid serving with provenance — retrieval for known, fine-tune for novel" (current default framing already supports this). If retrieval wins everywhere, ship as retrieval-only and demote the QLoRA adapter to "experimental comparator" in the credential narrative — MLflow runs + training script + adapter SHA still constitute a real technique credential. | M3 small-sweep loss curve plateaus near base-Qwen on validation; OR M4 eval shows retrieval >= fine-tune on out-of-domain slice |

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
| 2A 20-pair toy gold | ⚠ Drafted but **0/20 human-verified** (2026-05-25) — `data/gold/toy_v1.jsonl` exists and is schema-valid (`dataset_hash = 41250143…`); 10 pairs are STUBs (best-guess citation_id), 10 are claude-extract-only. See the banner at the top of [data/gold/toy_v1_provenance.md](../data/gold/toy_v1_provenance.md). | **M0 still blocked** until author verification pass completes; tier 3A *can* run mechanically against the draft but its numbers are against unvalidated gold and must not be cited |
| 2B eval harness | ✓ Done — citation match + Gemini judge; tested end-to-end | See §9.2 |
| 2C tokenizer audit | ✓ Done & PASS for th/fr/de/en — R4 resolved | See §9.2 |
| 2D parser bake-off | ✓ Done — Marker locked; R1 resolved | See §9.2 |
| 3A baselines on toy gold | ⏳ Blocked on 2A | Plan in §9.3 |
| 3B lock parser choice (README) | ⏳ Pending — bake-off result is in [data/parser_bakeoff/summary.md](../data/parser_bakeoff/summary.md); README still uses forward-tense phrasing | Plan in §9.3 |

### 9.1 — Tier 1 detail (what landed + 1C closure plan)

**1A — Repo skeleton + lockfile.** Layered uv-project layout, all gitignored venvs:

- **Root project** (`pyproject.toml` + `uv.lock` + `.venv/`) — pure-Python shared library + lightweight CLI scripts. Core deps: `pydantic>=2.13`, `mlflow>=2.18`, `httpx>=0.28`, `pyyaml>=6`. Dev tooling: `ruff` (lint + format, line-length 100), `pyright` (basic mode), `pytest`.
- **`src/daccord/`** — shared library; layout reflects pipeline stages:
  - [validation.py](../src/daccord/validation.py) — `@validated` decorator + `ValidatedModel` base class (pydantic `validate_call` wrapper); applied to every named function/method/class per [CLAUDE.md](../CLAUDE.md).
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
6. **Verification**: `conda run -n d-accord --no-capture-output uv run pyright` (catches Literal extension misses), `conda run -n d-accord --no-capture-output uv run pytest` (covers new cost tests), `conda run -n d-accord --no-capture-output uv run python -m daccord.costs status` (prints all four free-tier providers with `0/<cap>` usage on day-zero).

**1D — PDF corpus.** Idempotent download script + manifest; 13 PDFs across 8 jurisdictions on disk.

- Script: [scripts/download_corpus.py](../scripts/download_corpus.py) — flags `--sources`, `--raw-root`, `--manifest`, `--frameworks`, `--dry-run`; skips files whose SHA256 already matches the manifest. Supports URL-fetch (httpx) and manual placement (prints expected path + waits).
- Spec: [data/sources.yaml](../data/sources.yaml) — 13 source entries across 8 frameworks. 9 auto-fetch (EU GDPR, BDSG DE+EN, PDPA-TH English, PH DPA + IRR, PDPA-MY Act + Amendment, FR Loi I+L), 4 manual (UK-GDPR, UK DPA 2018, PDPA-SG, PDPA-TH Thai — per R8 the UK + FR sources fall back to browser-printed PDFs because legislation.gov.uk and Légifrance expose no scraper-friendly consolidated PDFs).
- Manifest: [data/raw_manifest.json](../data/raw_manifest.json) — list of `ManifestEntry(framework, jurisdiction, filename, sha256, fetched_at)`; auto-updated by the script.
- On-disk PDFs (13 files) under `data/raw/<jur>/<framework>/`: `de/bdsg/{bdsg_de_current,bdsg_en_current}.pdf`, `eu/gdpr/reg_2016_679_consolidated.pdf`, `fr/loi_il/loi_78_17_consolidated.pdf`, `my/pdpa_my/{pdpa_my_act709_bilingual,pdpa_my_amendment_act_a1727_2024}.pdf`, `ph/dpa_2012_ph/{dpa_2012_ph,dpa_2012_ph_irr_amended}.pdf`, `sg/pdpa_sg/pdpa_sg_current.pdf`, `th/pdpa_th/{pdpa_th_thai_2019,pdpa_th_english_2019}.pdf`, `uk/{dpa_2018,uk_gdpr}/*_current.pdf`.
- R8 follow-up at tier 4: spot-check Marker output on UK-GDPR / UK DPA 2018 / FR Loi I+L against a regulator-issued reference (e.g., BDSG); if unrecognised-token rate >2× the auto-downloaded baseline, switch to headless-browser PDF export with print-CSS suppression or source from Legifrance API / legislation.gov.uk Atom feed.

### 9.2 — Tier 2 detail (what landed + 2A creation plan)

**2A — 20-pair hand-built toy gold.** ⚠ **Drafted 2026-05-25 but 0/20 human-verified.** The artifact exists and is schema-valid; the *credential's "human-verified" bar* is not met yet. M0 cannot close until the author completes the per-pair PDF pass — see the banner at the top of [data/gold/toy_v1_provenance.md](../data/gold/toy_v1_provenance.md) for the load-bearing warning that future sessions need to see.

What landed:

- Gold file: [data/gold/toy_v1.jsonl](../data/gold/toy_v1.jsonl) — 20 rows, schema-valid, `dataset_hash = 412501438684f1ea9c2fcfbdcbb92897cb469fd795c61f8839e193824e3880a5`. The hash will change once the author edits any cell during the verification pass.
- Coverage matrix: [data/gold/toy_v1_coverage.md](../data/gold/toy_v1_coverage.md) — pre-plan of the 20 pairs against jurisdiction × concept × difficulty axes. Coverage gates all PASS mechanically (20 rows, all 8 jurisdictions ≥2, th=4 ≥3, fr=4 ≥3) per [envs/eval/scripts/verify_toy_coverage.py](../envs/eval/scripts/verify_toy_coverage.py). Mechanical pass; does not validate citation correctness.
- Provenance + double-check punch list: [data/gold/toy_v1_provenance.md](../data/gold/toy_v1_provenance.md) — every row currently `verified_by: UNVERIFIED`. Two tiers of follow-up: 10 STUB rows (best-guess citation_id, likely needs correction) + 10 claude-extract-only rows (citation_id confirmed via direct PDF extract, mechanism text needs author semantic pass).
- Drafter (unused this round): `envs/eval/scripts/draft_toy_gold.py` calls Llama 3.3-70B (Groq) + Gemini 2.5-Flash (Google AI Studio) and writes `data/gold/.draft_*.jsonl` (gitignored). This round used direct PDF text extraction via [envs/audit/scripts/extract_pdf_text.py](../envs/audit/scripts/extract_pdf_text.py) instead, skipping the LLM-hallucination layer for the 6 frameworks claude could read first-hand (GDPR, PDPA-SG, PDPA-TH English, PDPA-MY, DPA-PH, BDSG English).

What's still needed before M0 can close (per the 2A acceptance criteria below):

1. Author opens each `data/raw/<jur>/<framework>/` PDF; for each pair confirms (a) citation_id exists with that exact string, (b) `*_mechanism` paraphrase is faithful to actual text.
2. STUB rows additionally need the citation_id corrected, not just confirmed (FR Loi I+L post-2018 renumbering, BDSG Sec 38 threshold, UK DPA 2018 Part 3 sections, etc.).
3. `verified_by` + `verification_date` cells in the provenance table get populated row-by-row.
4. PENDING notes in the JSONL's `notes` field get removed once the row is verified (changing `dataset_hash`).
5. Re-run [verify_toy_coverage.py](../envs/eval/scripts/verify_toy_coverage.py) + `run_eval.py --dry-run` to confirm nothing broke during the verification edits.

**If a future session sees weird `eval/baseline_toy.csv` numbers on a TH / FR / DE / UK pair — check the provenance banner first.** If the relevant pair is still flagged STUB, the gold is the bug, not the model under eval.

**2A creation plan** — block tier 3A on this:

1. **Pre-plan the 20 pairs against a coverage matrix** before drafting. The matrix balances three axes simultaneously so the M0 baseline and the future M4 per-jurisdiction breakdown are interpretable:
   - **Jurisdiction coverage** — each of the 8 jurisdictions (EU, UK, DE, FR, SG, TH, PH, MY) must appear as source or target in ≥2 pairs.
   - **Native-language validation moat** — ≥3 pairs touch PDPA-TH (author validates Thai natively); ≥3 pairs touch FR Loi I+L (author validates French). These are the credential's headline claim; under-sampling them hollows out M4.
   - **Concept axes** (per the draft script) — consent, data subject access, breach notification, security obligations, DPO duties. Hit each axis ≥3 times across the 20 pairs.
   Sketch the matrix in `data/gold/toy_v1_coverage.md` (committed alongside the gold) so the choice is defensible.
2. **Run the drafter**: `cd envs/eval && conda run -n d-accord --no-capture-output uv run python scripts/draft_toy_gold.py`. Outputs gitignored.
3. **Reconcile** — for each target pair, open both drafts side-by-side. Pick the stronger, merge, or human-write. Llama/Gemini disagreement on `target_citation_id` is itself a signal of conceptual ambiguity → either sharpen the concept selection or accept as a deliberately-hard pair (useful for the eval set; documents that ambiguity exists in the source frameworks).
4. **Verify against the source PDF** — for every merged pair, open the relevant file in `data/raw/<jurisdiction>/<framework>/`:
   - Confirm the cited article/section exists in the PDF (exact `citation_id`, not paraphrased).
   - Confirm `target_mechanism` paraphrases the actual text (not the LLM's hallucinated gist).
   - For Thai PDPA pairs: cite from `pdpa_th_thai_2019.pdf` (authoritative) and cross-check `pdpa_th_english_2019.pdf`. Same pattern for BDSG (German authoritative).
   - For UK / FR pairs: be aware these PDFs are browser-print (R8 caveat) — header chrome may shift page numbers; cite by article/section, never by page.
5. **Commit** — write merged + verified pairs to `data/gold/toy_v1.jsonl`. Fill every row of the provenance table.
6. **Schema-validate at write time** — `cd envs/eval && conda run -n d-accord --no-capture-output uv run python scripts/run_eval.py --dry-run --gold-path ../../data/gold/toy_v1.jsonl --verbose`. Loads via `GoldSet.from_jsonl`, prints `dataset_hash`, builds a sample prompt; zero API calls.
7. **Acceptance criteria** (block 3A on these):
   - [ ] `data/gold/toy_v1.jsonl` exists and contains exactly 20 lines.
   - [ ] `--dry-run` loads cleanly and prints a stable `dataset_hash`.
   - [ ] All 8 jurisdictions appear ≥2 times (source or target).
   - [ ] PDPA-TH and FR Loi I+L each appear ≥3 times.
   - [ ] Provenance table is fully populated (no empty `verified_by` / `verification_date` cells).
   - [ ] `data/gold/toy_v1_coverage.md` documents the matrix.
8. **NOT in scope here**: scaling beyond 20 pairs (that's tier 9, gold freeze ≥500 pairs).

**2B — Eval harness.** Done; tested; runnable today.

- Sub-project: `envs/eval/` (own `pyproject.toml`, `uv.lock`, tests). Provider SDKs (Groq, Google GenAI) isolated here to keep the root venv slim.
- CLI: [envs/eval/scripts/run_eval.py](../envs/eval/scripts/run_eval.py). Flags: `--gold-path`, `--models` (comma-separated aliases — currently `groq,gemini`), `--judge` (default `gemini-2.5-flash`), `--output-csv`, `--run-name`, `--prompt-variant`, `--seed`, `--dry-run`.
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

Methodology + diagnostic tables (incl. top-fragmented Thai characters) in [eval/tokenizer_audit.md](../eval/tokenizer_audit.md); machine-readable in [eval/tokenizer_audit.csv](../eval/tokenizer_audit.csv); logged to MLflow run `6537272d8e7f4af0a72929ec6573a93d` under experiment `daccord-tokenizer-audit`. Reproduce: `cd envs/audit && conda run -n d-accord --no-capture-output uv run python scripts/run_tokenizer_audit.py`. Qwen2.5-7B-Instruct is locked as the base model.

**2D — Thai parser bake-off.** Done; **Marker locked**. R1 resolved.

5-page Thai sample (PDPA-TH 2019, pages [1, 11, 16, 39, 42]) scored against 61 hand-verified Thai citations (`มาตรา N`). Both parsers hit perfect recall + precision; reading-order is the tie-breaker.

| Parser | Recall | Precision | Reading order (Thai-reader judgment) | Structure preserved |
|---|---:|---:|---:|---:|
| marker | 1.000 | 1.000 | 5.000 | 1.000 |
| typhoon | 1.000 | 1.000 | 4.000 | 1.000 |

Full report: [data/parser_bakeoff/summary.md](../data/parser_bakeoff/summary.md). Decision applies to tier 4 (full-corpus parse to markdown) for both EN and TH. R1 cut criterion not triggered.

### 9.3 — Tier 3 detailed plan

**Hard prerequisite gate**: §9.2's 2A creation plan must be complete — `data/gold/toy_v1.jsonl` committed and `--dry-run` clean — before any 3A step runs. 3B is parallelisable with 2A.

#### 3A — Baselines on toy gold

Run three generators on the 20-pair toy gold, judged by Gemini 2.5-Flash, and emit `eval/baseline_toy.csv` + MLflow run history. The three baselines are: **base Qwen 2.5-7B-Instruct (local, 4-bit NF4)** + **Llama 3.x 70B via Groq** + **Gemini 2.5-Flash via Google AI Studio**.

The Qwen baseline runs at **4-bit NF4 quantization** (the same precision the tier 10–12 QLoRA training will load at), not bf16 — this is a deliberate choice so the M4 fine-tune-delta is apples-to-apples with the actual production load condition, not against a stronger-than-shipped baseline.

1. **Scaffold `envs/baseline/` sub-project** (new). Mirrors `envs/eval/` / `envs/audit/` / `envs/bakeoff/`:
   - `envs/baseline/pyproject.toml` — declare `daccord` via `[tool.uv.sources]` path source `../../`. Pin (after verifying current stable per the latest-stable convention — search web before committing pins): `transformers>=5.9,<6` (match `envs/audit/`), `torch>=2.5,<3` (CUDA 12.x build for RTX 5080 Blackwell), `bitsandbytes>=0.45,<0.46`, `accelerate>=1.2,<2`.
   - `envs/baseline/.venv/` gitignored via existing `envs/*/.venv/` rule.
   - Sync: `conda run -n d-accord --no-capture-output uv --project envs/baseline sync`.
2. **Add `LocalHFClient` to [src/daccord/eval/clients.py](../src/daccord/eval/clients.py)** implementing the existing `ModelClient` Protocol:
   - `provider: Provider = "local_hf"` — extend the `Provider` Literal in [src/daccord/costs/config.py](../src/daccord/costs/config.py) to include `"local_hf"` (separate edit from §9.1's 1C closure). Classify as free-tier with a symbolic high RPD cap (or no cap). `estimate_cost` returns 0.0; `record_call` still runs so MLflow gets latency + token counts.
   - `__init__(model="Qwen/Qwen2.5-7B-Instruct", quantization="nf4")` — lazy-import torch + transformers + bitsandbytes; load with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)`; cache model on instance.
   - `generate(...)` — apply the Qwen chat template; greedy decode (`temperature=0, do_sample=False`) to match `GroqClient` / `GeminiClient`; constrain output via the existing `_CANDIDATE_JSON_SCHEMA` (best effort through the prompt — Qwen has no native JSON-schema constraint); reuse `_parse_candidate` for failure recovery (parse failures record as Tier-1 misses with the error in `judge_reasoning`).
   - Decorate every public method with `@validated`.
   - Tests at `envs/baseline/tests/test_local_hf_client.py` — mock `from_pretrained` with a small stand-in; verify `generate` returns a well-formed `ModelResponse`. CI-friendly without GPU.
3. **Wire alias `qwen` into [envs/eval/scripts/run_eval.py](../envs/eval/scripts/run_eval.py)**:
   - Add `"qwen": "Qwen/Qwen2.5-7B-Instruct"` to `MODEL_ALIASES`.
   - Add a `qwen` branch in `_resolve_generators` instantiating `LocalHFClient(model=...)`. One elif; no other runner changes (the Protocol does the rest).
   - **Run with the `envs/baseline/` venv** when `qwen` is in `--models` — eval scripts physically live under `envs/eval/scripts/` but execute against `envs/baseline/`'s venv so heavy torch/bnb deps stay localised. Both venvs install `daccord` via path source, so the import surface is identical.
4. **Execute baseline run**:
   ```
   cd envs/baseline && conda run -n d-accord --no-capture-output uv run ^
     python ..\eval\scripts\run_eval.py ^
     --gold-path ..\..\data\gold\toy_v1.jsonl ^
     --models qwen,groq,gemini ^
     --judge gemini-2.5-flash ^
     --output-csv ..\..\eval\baseline_toy.csv ^
     --run-name baseline-toy-YYYY-MM-DD ^
     --seed 42
   ```
   Expected outputs: `eval/baseline_toy.csv` with 60 rows (20 pairs × 3 models); MLflow parent run `baseline-toy-YYYY-MM-DD` under experiment `daccord-eval`; three nested child runs (one per generator) with `tier1_citation_match_overall`, per-jurisdiction breakdowns, `tier2_judge_mean`, judge-bucket counts.
5. **Update [eval/README.md](../eval/README.md)** with M0 baseline numbers and a pointer to `eval/baseline_toy.csv`. Flag any baseline that scores >0.7 on Tier-1 — that becomes the credible-delta floor the QLoRA fine-tune (M4) must beat to clear the M4 cut criterion (§4).

#### 3B — Lock parser choice (README finalize)

No code. Documentation closure of the M1 parser-bakeoff artifact.

- Update [README.md](../README.md) Architecture section (around the `data/ingest/` row) and Tech Stack PDF processing entry. They currently reference the bake-off in forward tense ("locked for both EN and TH after the tier-2D bake-off"); convert to past-tense numeric: *"Marker locked for both EN and TH (Thai bake-off: marker recall=1.0 precision=1.0 reading-order=5.0 vs typhoon reading-order=4.0; see [data/parser_bakeoff/summary.md](data/parser_bakeoff/summary.md))."*
- Keep the public README free of commercial/competitive language. "Marker locked" is a factual technical statement; that's the entire content of the change.

#### M0 closure checklist

- [ ] `data/gold/toy_v1.jsonl` committed with 20 verified pairs (closes 2A)
- [ ] `eval/baseline_toy.csv` committed (3A artifact)
- [ ] MLflow `daccord-eval` experiment has the `baseline-toy-*` parent run with 3 child runs
- [ ] README parser-bakeoff section is past-tense with numeric outcome (3B)
- [ ] `eval/tokenizer_audit.md` already committed — closes 2C ✓ done
- [ ] `data/parser_bakeoff/summary.md` already committed — closes 2D ✓ done
- [ ] Costs config covers all 4 free-tier providers (closes 1C; required before tier 7A, not strictly before M0)

#### Verification commands (for 3A execution)

```
conda run -n d-accord --no-capture-output uv lock --check
conda run -n d-accord --no-capture-output uv run ruff check .
conda run -n d-accord --no-capture-output uv run ruff format --check .
conda run -n d-accord --no-capture-output uv run pyright
conda run -n d-accord --no-capture-output uv run pytest
conda run -n d-accord --no-capture-output uv --project envs/baseline lock --check
cd envs/baseline && conda run -n d-accord --no-capture-output uv run pytest
```

Plus the baseline command above (live run, hits Groq + Gemini free tiers).

---

## Critical files

- [aws_credential_plan.md](../aws_credential_plan.md) — authoritative architecture
- [README.md](../README.md) — parser-bakeoff rationale + eval results updates land here after M4
- `eval/run_eval.py` (to be created — the M0 eval harness is the project's first hard gate)
- `training/train.py` (to be created — HF `transformers` + `peft` + `bitsandbytes` + `trl` stack; small-sweep first at M3)
- `scripts/teardown_endpoint.py` (to be created — committed before Phase 2 first stand-up)
- `scripts/deploy_endpoint.py` (to be created — Phase 2 re-stand-up on demand)
