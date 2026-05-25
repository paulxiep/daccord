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

**Effort baseline**: ~3 wk (~0.5–0.7 EM). AWS cost ceiling **$50–100**.

### Phased execution

- **Phase 1 — Local validation (M0–M4)**: full data pipeline + QLoRA training on RTX 5080 + three-tier eval. **All deliverables except the SageMaker endpoint close here.** No AWS-runtime dependency, no AWS spend risk.
- **Phase 2 — SageMaker hosting (M5)**: deploy adapter to endpoint via boto3, smoke test, capture, tear down. Decoupled from Phase 1; can be triggered when a concrete demo opportunity justifies the $50–100 spend.
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

### Phase 1 — Local validation

| Tier | Tasks | Type | Notes |
|---|---|---|---|
| **1** | 1A repo skeleton + lockfile · 1B MLflow + autolog plumbing · 1C per-provider RPD caps for free-tier APIs (Groq/Cerebras/Google AI Studio/DeepSeek) · 1D start PDF corpus download | parallel | All independent; day-1 |
| **2** | 2A 20-pair hand-built toy gold · 2B eval harness (citation match + judge) · 2C tokenizer audit (Thai/FR/DE) · 2D Thai parser bake-off on 5-page sample (Marker vs Typhoon-OCR — **Marker locked**, see `data/parser_bakeoff/summary.md`) | parallel | No full corpus needed |
| **3** | 3A baselines on toy gold (base Qwen 3-8B + Llama 4 Scout via Groq + Gemini 3.1 Flash Lite via Google AI Studio) · 3B lock parser choice from 2D results | parallel | 3A needs 2A+2B; 3B needs 2D; baselines are all free-tier OSS to match the tier 7A ensemble |
|  | **[M0 gate]** | | tokenizer passes · baselines captured · parser locked |
| **4** | Parse all PDFs to markdown (Marker, locked for both EN and TH) | sequential | Needs 1D complete + 3B parser choice. **Watch R8**: 3 sources (UK-GDPR, UK DPA 2018, FR Loi I+L) come from browser print-to-PDF (Légifrance/legislation.gov.uk expose no scraper-friendly consolidated PDF) — 5–60× larger than regulator-issued PDFs, layout may confuse Marker |
| **5** | Citation registry extraction per framework | sequential | |
|  | **[M1 gate]** | | corpus + registries frozen |
| **6** | 6A ensemble prompt + JSON schema (citations constrained to registry from 5) · 6B tiering script | parallel | |
| **7** | 7A ensemble generation — 4-model OSS via free-tier APIs (Llama 4 Scout/Groq + Qwen 3-32B/Groq or Cerebras + Gemini 3.1 Flash Lite/Google AI Studio + DeepSeek V3) (~3d **async**) · 7B splits script · 7C hand-validate completed framework-pairs as they land | parallel | 7A async pacing now driven by free-tier RPD limits, not API spend; 7B/7C fill the wait |
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

All ML substance happens here. No AWS resources stood up. Zero AWS spend risk. If Phase 2 is deferred indefinitely, Phase 1 still constitutes a complete deliverable set (eval CSV + MLflow runs + local-inference recording + README).

### M0 — Eval Bar Locked (end of d3)

- **DoD**: 20-pair toy gold built · eval harness runs end-to-end · baselines captured (base Qwen 3-8B + Llama 4 Scout via Groq + Gemini 3.1 Flash Lite via Google AI Studio) on toy · tokenizer audit committed
- **Artifact**: `eval/baseline_toy.csv` + `eval/tokenizer_audit.md`
- **Cut criterion**: tokenizer audit shows Qwen3-8B fragments Thai at >2 tokens/char average → escalate immediately: swap base (SeaLLM-v3, Typhoon-7B) or descope Thai. **Decide here, not at training time.**

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
- **Cut criterion**: fine-tune delta vs base Qwen <5% on Tier-1 citation accuracy AND negative vs Llama 3.x 70B on every jurisdiction AND **no advantage over the retrieval baseline on the out-of-domain slice** → **do not push to SageMaker**. The retrieval-baseline qualifier is what makes the cut honest: if retrieval ties or beats fine-tune everywhere, ship as retrieval-only (architecture pivot), don't ship the heavier serving stack just to preserve the original framing. Document the honest negative result in the eval CSV. (No paid-API spend to tear down — ensemble + baselines + judge are all free-tier OSS.)

### Phase 2 — SageMaker hosting (M5)

Trigger when M4 has a publishable delta AND there's a concrete reason (demo, runway) to absorb the AWS spend. Until then, Phase 1 artifacts are sufficient.

### M5 — Endpoint Live, Captured, Torn Down (~2–3 days when triggered)

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

- **Phase 1 spend** is effectively $0 — ensemble (7A) + LLM-as-judge (13) both run on open-source models via free-tier APIs (Groq, Cerebras, Google AI Studio, DeepSeek direct). Only paid Phase 1 cost is ~$5–10 LlamaParse fallback if Marker fails on a specific document. One row per day in `costs/daily.csv` committed to repo with **request-count** entries against per-provider RPD caps (Groq ~14400 RPD, Google AI Studio ~1500 RPD; Cerebras + DeepSeek quotas verified at run time). Hard $5/provider paid-spill ceiling if free tier exhausts during a run.
- **Phase 2 SageMaker discipline**: `ml.g5.xlarge` ≈ $1.40/hr; **target <48 h total live**; stand up → smoke test (10 prompts) → capture → tear down (~2 h live). Re-stand-up on demand from `scripts/deploy_endpoint.py`; budget for ~5–10 min cold start for live demos. Cold start now also loads the MPNet embedder + FAISS index alongside the 7B adapter (~1–2 GB additional read; negligible time impact vs the adapter load).
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
| R4 | Qwen3-8B tokenizer fragments Thai/FR worse than expected | Low-Medium | High (kills language-validation moat) | M0 tokenizer audit before training plumbing; swap to SeaLLM-v3 or Typhoon-7B if Thai byte-fallback >20% | Audit shows >2.5 tokens/char on Thai or byte-fallback artifacts |
| R5 | RTX 5080 OOM at QLoRA-7B with full seq_len | Medium | Medium (slows training) | M3 small-sweep catches before full train; mitigations: max_seq_len 2048, gradient checkpointing, micro-batch 1 + grad-accum 16; **swap to Unsloth** if needed | Small-sweep OOMs at any seq_len >1024 |
| R6 | Fine-tune delta vs frontier baseline marginal or negative | Medium | High (kills the headline value claim — "small specialist beats / matches frontier") | M0 baselines set expectation early; M4 cut = document honest negative result and skip Phase 2 | Tier-1 citation accuracy delta vs base Qwen <5% by mid-train checkpoint |
| R7 | Free-tier RPD exhaustion stalls ensemble (Phase 1 spend is $0 OSS-via-free-tier; risk is rate, not cost) | Low | Low (delays, not $ ceiling breach) | Per-provider RPD caps + checkpointed ensemble outputs; runner waits for daily reset and resumes; hard $5/provider paid-spill ceiling if absolutely needed | Free-tier RPD exhaustion on 2 providers within same eval-pass window |
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

---

## Critical files

- Internal architecture plan (gitignored) — authoritative architecture (Pillar B SageMaker hosting)
- [README.md](../README.md) — parser-bakeoff rationale + eval results updates land here after M4
- `eval/run_eval.py` (to be created — the M0 eval harness is the project's first hard gate)
- `training/train.py` (to be created — HF `transformers` + `peft` + `bitsandbytes` + `trl` stack; small-sweep first at M3)
- `scripts/teardown_endpoint.py` (to be created — committed before Phase 2 first stand-up)
- `scripts/deploy_endpoint.py` (to be created — Phase 2 re-stand-up on demand)
