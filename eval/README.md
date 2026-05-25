# `eval/` — three-tier eval harness

Tier 2B deliverable. The harness reused at M0 (toy gold, baselines) and M4 (full gold, post-train results) — same code, same CSV row contract, same MLflow shape, different input.

> ⚠ **Gold-set status (2026-05-25)**: [data/gold/toy_v1.jsonl](../data/gold/toy_v1.jsonl) is an **unvalidated draft** — 0 / 20 pairs human-verified, 10 STUB rows with best-guess citation_ids. Running this harness against the toy gold today *will* produce `eval/baseline_toy.csv` numbers, but they are against unvalidated gold and **must not be cited as M0 baselines**. M0 stays open until [data/gold/toy_v1_provenance.md](../data/gold/toy_v1_provenance.md) drops the banner at its top. See [docs/development_plan.md §9.2](../docs/development_plan.md) for the 2A closure plan.

## CSV row contract

Stable from M0 → M4. Future readers (M4 README table, downstream analysis notebooks) parse this format:

```
gold_id, model, source_jurisdiction, source_framework,
target_jurisdiction, target_framework, source_language, target_language,
predicted_citation_id, expected_citation_id,
citation_match, judge_score, judge_bucket, judge_reasoning
```

| Column | Type | Notes |
|---|---|---|
| `citation_match` | 0 / 1 | Tier-1: normalized exact match (see [src/daccord/eval/scoring.py](../src/daccord/eval/scoring.py) `normalize_citation_id`). |
| `judge_score` | float [0, 1] | Tier-2: LLM-as-judge continuous score. |
| `judge_bucket` | enum | `wrong` / `partial_wrong` / `partial_right` / `substantively_right` / `exact`. |
| `judge_reasoning` | str | One sentence — seed for M4 Tier-3 human spot-check. |

Per-jurisdiction, per-language, and per-framework-pair aggregates are **NOT** in the CSV — they live in MLflow child-run metrics (see below). The CSV is per-pair, per-model only; aggregation happens at read time so a single source of truth (the rows) drives every breakdown.

## MLflow shape

One experiment `daccord-eval`. Each `run_eval` invocation produces a parent run + one nested child per generator model:

```
experiment: daccord-eval
└── parent run: "baseline-toy-YYYY-MM-DD"
    tags: project=d-accord, gate=M0, prompt_variant=..., judge_model=...
    params: dataset_hash, git_commit, seed, judge_model, n_gold_pairs, n_generators
    artifact: <output_csv>
    │
    ├── child run: ".../groq"  or  ".../<model-id>"  (nested=True)
    │   params: model, provider, judge_model, dataset_hash, prompt_variant, seed
    │   metrics:
    │     tier1_citation_match_overall
    │     tier1_citation_match__jur__<jurisdiction>         (per-target-jurisdiction)
    │     tier1_citation_match__lang__<lang>                (per-target-language)
    │     tier1_citation_match__fwpair__<src>__<tgt>        (per-framework-pair)
    │     tier2_judge_mean
    │     tier2_judge_pct_above_0_7
    │     judge_bucket_<bucket>                              (count, per bucket)
    │     n_pairs
    │
    └── child run: ".../gemini-3.1-flash-lite"  (same metric schema; default judge is meta-llama/llama-4-scout-17b-16e-instruct)
```

`prompt_variant` is a REQUIRED runner argument (no silent default). At M0 it's `"unconstrained-m0"`; tier 6A's registry-constrained prompt will land a different variant string.

## Generator + judge providers (free-tier)

Wired at 2B (project pivoted to free-tier OSS for Phase 1):

| Alias | SDK | Default model | Cap mode |
|---|---|---|---|
| `groq` | [groq](https://pypi.org/project/groq/) | `meta-llama/llama-4-scout-17b-16e-instruct` | RPD (free tier) |
| `gemini` | [google-genai](https://pypi.org/project/google-genai/) | `gemini-3.1-flash-lite` | RPD (free tier) |
| `retrieval` | [sentence-transformers](https://pypi.org/project/sentence-transformers/) + [faiss-cpu](https://pypi.org/project/faiss-cpu/) | `paraphrase-multilingual-mpnet-base-v2` (embedder) | n/a — local-only, no API spend |

API keys come from `.env.local` (`GROQ_API_KEY`, `GOOGLE_API_KEY`). Every call routes through [daccord.costs](../src/daccord/costs/__init__.py) (`preflight` + `record_call`) — free-tier providers raise `CapExceeded` on RPD-cap breach; paid-spill providers raise on USD-cap breach. The `retrieval` client is local-only and bypasses the cost layer.

## Retrieval baseline

Fourth comparator added at tier 12B. Answers the architectural question *"could you have just done retrieval over the gold set?"* with data, not hand-waving.

**How it works**: embed each train-split source clause with the multilingual MPNet embedder; index with FAISS. At eval time, embed the val/test source clause, retrieve top-1 by cosine, return the indexed gold pair's `target_mechanism` + `citation_id` + `mapping_justification` verbatim. Zero LM call, zero citation hallucination by construction.

**Why it's there**: the M4 cut criterion broadens to `fine-tune delta vs base Qwen <5% AND negative vs Llama 70B on every jurisdiction AND no advantage over retrieval on the out-of-domain slice → do not push to SageMaker`. The retrieval comparator is what makes the third clause measurable. It also feeds back into serving — the same FAISS index is reused at inference time by `HybridRouter` in `src/daccord/serving/hybrid.py` (retrieval-first, QLoRA fallback).

**Stratification**: each eval pass is run twice with `--slice-tag in-domain` and `--slice-tag out-of-domain`. The tag goes to MLflow run tag + child-run param; **CSV row shape is unchanged** (slice is run-level metadata, not per-row, per the contract above). Two CSV files (or one with `--run-name` distinguishing them) cover both slices.

**Index build**: `cd envs/eval && uv run python scripts/build_retrieval_index.py --gold-path ../../data/splits/train.jsonl --output ../../data/indices/retrieval__train__<dataset_hash>.faiss`. Run once per train-split version (re-run when the split refreshes).

The local-HF baseline for base Qwen3-8B (tier 3A baseline run) is shipped via `LocalHFClient` in [src/daccord/eval/clients.py](../src/daccord/eval/clients.py). `torch` + `bitsandbytes` live in `envs/baseline/` (not the eval env), and Groq does not host the local-Qwen base — so the qwen alias routes through the `baseline` compose service for the GPU passthrough.

## Judge defaults + self-judging caveat

Default judge: `meta-llama/llama-4-scout-17b-16e-instruct` via Groq (bumped 2026-05-25 from `llama-3.3-70b-versatile` per the README "stronger judge" decision). `--judge gemini-3.1-flash-lite` is wired as an alternative for when a non-Llama judge is preferred (e.g., M4 with a Llama generator in the pool). When the generator and the judge are the same model id (e.g., `groq` generator + Llama 4 Scout judge), the judge is technically self-judging — a noise term at M0 (20 pairs) but a credibility risk at M4 (500 pairs); swap to DeepSeek V3 or another family then.

## CLI

The eval harness has its own sub-project venv at [envs/eval/](../envs/eval/) (mirroring tier 2D's [envs/bakeoff/](../envs/bakeoff/)) so its provider SDKs (`groq`, `google-genai`) don't leak into the shared root venv. Deps live in [envs/eval/pyproject.toml](../envs/eval/pyproject.toml); the CLI script in [envs/eval/scripts/run_eval.py](../envs/eval/scripts/run_eval.py).

All commands run from the project root. First-time setup: `uv --project envs/eval sync`.

```bash
# First-time provisioning (creates envs/eval/.venv + envs/eval/uv.lock)
uv --project envs/eval sync

# Dry-run — validates schema + builds prompts, no API calls
cd envs/eval && uv run python scripts/run_eval.py \
  --gold-path ../../data/gold/toy_v1.jsonl --dry-run

# Live M0 baseline against the toy gold
cd envs/eval && uv run python scripts/run_eval.py \
  --gold-path ../../data/gold/toy_v1.jsonl \
  --models groq,gemini \
  --judge gemini-3.1-flash-lite \
  --output-csv ../../eval/baseline_toy.csv \
  --run-name baseline-toy-2026-05-25

# Eval with retrieval baseline (requires index built via build_retrieval_index.py first)
cd envs/eval && uv run python scripts/run_eval.py \
  --gold-path ../../data/splits/test.jsonl \
  --models groq,gemini,retrieval \
  --retrieval-index-path ../../data/indices/retrieval__train__<dataset_hash>.faiss \
  --retrieval-embedder paraphrase-multilingual-mpnet-base-v2 \
  --slice-tag out-of-domain \
  --judge gemini-3.1-flash-lite \
  --output-csv ../../eval/results_v1__out_of_domain.csv \
  --run-name results-v1-out-of-domain-2026-05-25

# Run eval tests
cd envs/eval && uv run pytest

# View MLflow runs (mlflow lives in the root venv — invoke from there)
uv run mlflow ui --backend-store-uri file:./mlruns
# → http://127.0.0.1:5000, experiment "daccord-eval"
```

The CLI script defaults `--gold-path` and `--output-csv` to repo-root paths (via `REPO_ROOT = Path(__file__).resolve().parents[3]`), so the explicit `../../` arguments are only needed when you override defaults relative to the cwd inside `envs/eval/`.

## What's committed

| Path | Committed? | Purpose |
|---|---|---|
| `envs/eval/pyproject.toml` | yes | env manifest (groq + google-genai + sentence-transformers + faiss-cpu pins, daccord editable) |
| `envs/eval/uv.lock` | yes | env-local lockfile |
| `envs/eval/scripts/run_eval.py` | yes | CLI |
| `envs/eval/scripts/build_retrieval_index.py` | yes | retrieval-baseline index builder (tier 12B) |
| `envs/eval/tests/test_eval_*.py` | yes | eval-tier tests (run inside envs/eval venv) |
| `envs/eval/tests/test_retrieval_client.py` | yes | retrieval baseline unit tests |
| `eval/README.md` | yes | this file |
| `eval/baseline_toy.csv` | yes (after M0) | M0 baseline artifact |
| `eval/results_v1.csv` (+ `results_v1__out_of_domain.csv`) | yes (after M4) | M4 post-train results, stratified by slice |
| `data/indices/retrieval__train__<hash>.faiss` (+ parallel JSONL) | yes (after tier 12B) | FAISS index over train-split source clauses; reused at serving time by `HybridRouter` |
| `eval/traces/`, `eval/raw/` | no (gitignored) | per-run raw provider output (debug) |

The committed CSVs are the durable artifacts — every line traces back to its MLflow `dataset_hash` param, which traces back to a specific gold JSONL snapshot.

## Layering

- `daccord.gold` — pipeline-spanning gold-set shapes (`GoldPair`, `GoldSet`). Consumed by 2B/7A/9/10A. **Not** part of the eval namespace.
- `daccord.eval.schema` — inference-time shapes (`CitationCandidate`, `ModelResponse`, `PromptMessages`).
- `daccord.eval.prompts` — eval + judge prompt builders. Sibling `build_ensemble_prompt` lands here at tier 6A.
- `daccord.eval.clients` — provider adapters + `ModelClient` Protocol.
- `daccord.eval.scoring` — citation normalization, Tier-1 + Tier-2 + aggregation, `JudgeClient` Protocol + `GeminiJudge`.
- `daccord.eval.runner` — end-to-end orchestration + MLflow nesting + CSV writer.
