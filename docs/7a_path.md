# Tier 7A — Three execution paths

Tier 7A (ensemble label generation) has three concrete execution paths
behind a single `EnsembleStrategy` abstraction. Switching paths is one
CLI flag — `scripts/run_ensemble.py --strategy {bedrock-batch,
bedrock-sync, local-api-paid, local-api-free-ec2}`. All three produce
the same `data/ensemble/raw/{framework_pair}__{model}.jsonl` output
that tier 6B ([scripts/tier_ensemble.py](../scripts/tier_ensemble.py))
consumes unchanged.

This file is the durable trade-off record so future sessions don't
re-derive the numbers from scratch.

## Sizing (shared across all paths)

Clauses on disk today ([data/clauses/](../data/clauses/)):

| Framework | Clauses extracted | Registry IDs |
|---|---:|---:|
| gdpr | 74 | 92 |
| uk_gdpr | 80 | 103 |
| dpa_2018 | 204 | 288 |
| bdsg | 85 | 97 |
| loi_il | 123 | 126 |
| pdpa_sg | 28 | 91 |
| pdpa_th | 96 | 96 |
| pdpa_my | 146 | 146 |
| dpa_2012_ph | 44 | 72 |
| **Total registry** | — | **1,011** |

`build_prompts_for_pair()` in [scripts/run_ensemble.py](../scripts/run_ensemble.py)
iterates the **source registry** (not the clauses file), so prompts per
ordered pair = source registry size. Then × 8 targets per source × 4 seats:

- **Scope A** — full 72 pairs, no clause cap: ~8,088 prompts × 4 seats = **~32K invocations**
- **Scope B** — `--max-clauses-per-pair 30`: ~2,160 prompts × 4 seats = **~8.6K invocations**

Per call: **~485 input + ~256 output tokens ≈ 750 total** (smoke-run observation + prompt-builder math).

Tier-9 gold floor is ≥500 pairs. Scope B's 8.6K candidates is already 17×
that floor, so the **recommended first pass is Scope B on whichever path
the operator picks**.

---

## Path 1 — Bedrock batch (existing, blocked)

**Status**: blocked on AWS $0 spend limit; preserved (not deleted) so it's usable when AWS unblocks.

**Lineup** (F9-E, from [src/daccord/aws/m2.py](../src/daccord/aws/m2.py) `F9_BEDROCK_MODELS`):

| Seat | Model ID | Family |
|---|---|---|
| 1 | `meta.llama4-scout-17b-instruct-v1:0` | Meta |
| 2 | `meta.llama4-maverick-17b-instruct-v1:0` | Meta |
| 3 | `anthropic.claude-haiku-4-5-20251001-v1:0` | Anthropic |
| 4 | `amazon.nova-2-lite-v1:0` | Amazon |

**Cost** (Bedrock batch — 50% off on-demand from [costs/config.toml](../costs/config.toml)):
- Scope B: ~**$8–12** total
- Scope A: ~**$30–45** total

**Wall-clock**: 1–24 h cloud-side (Bedrock batch SLA).

**Code surface** (already shipped):
- [src/daccord/aws/batch.py](../src/daccord/aws/batch.py) — submit + poll + parse per model
- [src/daccord/aws/m2.py](../src/daccord/aws/m2.py) — region/role/bucket/model constants
- [scripts/run_ensemble.py](../scripts/run_ensemble.py) — `cmd_submit` / `cmd_poll` / `cmd_status` / `cmd_run_sync`
- Tests: [tests/test_bedrock_batch.py](../tests/test_bedrock_batch.py) + [tests/test_aws_m2.py](../tests/test_aws_m2.py)

**Modularization plan**: the code moves behind two `EnsembleStrategy`
impls (`BedrockBatchStrategy` + `BedrockSyncStrategy`) in a new
`src/daccord/aws/strategy.py`. `BatchPrompt` + `model_slug` hoist out of
`aws/batch.py` to `src/daccord/ensemble/prompt.py` with a back-compat
re-export so existing tests stay green.

---

## Path 2 — Paid direct API (label-quality, fast) — **recommended default**

**Lineup** — four 2025+ generation small-variant labelers, one per family (constraint: 2025+ generation, generation matters more than parameter count):

| Seat | Model ID | Family | $/M in | $/M out | Tier-1 RPM | Tier-1 RPD | Tier-1 TPM | Auto-tier-up |
|---|---|---|---|---|---|---|---|---|
| 1 | `claude-haiku-4-5` | Anthropic | $1.00 | $5.00 | **50** | none | 50K ITPM / 10K OTPM (cached reads free) | Auto → T2 (1000 RPM) at $40 cumulative spend |
| 2 | `gpt-5-mini` | OpenAI | $0.25 | $2.00 | **500** | none documented | 500K TPM | Auto → T2 at $50 cumulative + 7 days |
| 3 | `gemini-3.1-flash-lite` | Google | $0.25 | $1.50 | **4000** (operator's actual paid-tier quota) | **150,000** (operator's actual paid-tier quota) | 1M | Auto → T2 at $100 cumulative + 3 days |
| 4 | `Qwen/Qwen3-235B-A22B-Instruct-2507-tput` via Together | Alibaba | $0.20 | $0.60 | dynamic (concurrency-based; serverless tier) | none | dynamic | No tiers — auto-adjusts with sustained traffic |

> **2026-05-27 swap**: original pick was Llama 4 Maverick FP8 on Together; live API probe confirmed BOTH `Llama-4-Maverick-17B-128E-Instruct-FP8` AND `Llama-4-Scout-17B-16E-Instruct` are dedicated-endpoint-only (require paying for an always-on `g6e`-class deployment). Swapped to **Qwen 3-235B-A22B Instruct (2025-07 release)** which is serverless on Together at $0.20/$0.60 per 1M — cheaper than Maverick FP8 and same 2025+ generation. Family diversity now spans Anthropic / OpenAI / Google / Alibaba (one each).

**Rate-limit sources** (verified May 2026):
- Anthropic: [platform.claude.com/docs/en/api/rate-limits](https://platform.claude.com/docs/en/api/rate-limits)
- OpenAI: [GPT-5-mini model page](https://developers.openai.com/api/docs/models/gpt-5-mini), [community rate-limit thread](https://community.openai.com/t/increased-gpt-5-and-gpt-5-mini-rate-limits/1357840)
- Gemini: [ai.google.dev rate-limits](https://ai.google.dev/gemini-api/docs/rate-limits) (defers to AI Studio dashboard — numbers above are the gemini-2.5-flash-lite T1 floor, widely reported as carrying over)
- Together: [docs.together.ai rate-limits](https://docs.together.ai/docs/rate-limits)

**Gemini RPD risk — resolved**: this account's actual Gemini paid quota is 4000 RPM / 150K RPD (operator-confirmed in AI Studio). Gemini is no longer the bottleneck at any scope; Anthropic Haiku 4.5 at 50 RPM (Tier 1) is the wall-clock floor until $40 cumulative spend auto-promotes to 1000 RPM.

**Runtime RPM override**: the per-provider throttle in [src/daccord/eval/_rpm.py](../src/daccord/eval/_rpm.py) reads `DACCORD_RPM_<PROVIDER>` env vars at call time (e.g. `DACCORD_RPM_GOOGLE_GEMINI=4000`) so operators on paid tiers can lift the conservative default without code changes. Defaults stay at the free-tier ceiling so unmodified code remains safe.

**Cost**:

| Seat | Scope B | Scope A |
|---|---:|---:|
| Haiku 4.5 | $3.67 | $14.13 |
| GPT-5-mini | $1.32 | $5.07 |
| Gemini 3.1 Flash Lite | $1.05 | $4.05 |
| Qwen 3-235B (Together) | $0.52 | $2.02 |
| **Total** | **~$6.56** | **~$25.27** |

**Wall-clock** (in parallel, one thread per provider; bottleneck = slowest seat):

**Important correction (2026-05-27 from a partial live run)**: with one sequential worker per seat, throughput is **latency-bound**, not RPM-bound. Per-call latency measurements:

| Seat | Per-call latency | Sequential throughput |
|---|---|---|
| Anthropic Haiku 4.5 | ~3.3 s | ~18 RPM |
| Gemini 3.1 Flash Lite | ~3.9 s | ~15 RPM |
| Qwen 3-235B (Together) | ~3-4 s | ~15-20 RPM |
| GPT-5-mini, `reasoning_effort="minimal"` | ~3.0 s | ~20 RPM |
| GPT-5-mini, default reasoning | ~13-15 s | ~4 RPM |

With the `reasoning_effort="minimal"` fix (this MR), all 4 seats are in the 3-4 s/call range. **Each seat takes ~ N_prompts × 3.5 s** before parallelism across seats:

- Scope B (2160 prompts per seat): ~2 h
- Scope A (8888 prompts per seat): **~9 h**

The original "~45 min / ~2.7 h" estimates assumed Anthropic 50 RPM would be the bottleneck — but that's only reached when N concurrent workers per seat fire calls simultaneously. With sequential per-seat workers, server-side compute dominates. To push the Tier-1 RPM ceiling, see Path 4's "what Path 4 is NOT solving" note (per-seat concurrency is a separate refactor).

Resilience contract still gives clean resume on any interruption; operator-time is `<1 min` to launch + check completion.

**Code surface** (new):
- New clients in [src/daccord/eval/clients.py](../src/daccord/eval/clients.py): `AnthropicClient`, `OpenAIClient`, `TogetherClient` (Gemini already wired — just point at the 3.1 Flash Lite ID).
- New strategy: `LocalAPIStrategy` in `src/daccord/ensemble/local_api.py`.
- Per-provider throttle rewrite in [src/daccord/eval/_rpm.py](../src/daccord/eval/_rpm.py) (see Shared Infrastructure below).
- Pricing rows + USD caps in [costs/config.toml](../costs/config.toml).

---

## Path 3 — Free API ensemble across 4 providers (cheapest $, RPD-bound)

**Status**: budget-zero option. Use when paid budget is tight and we can absorb a multi-day wall-clock from free-tier RPD ceilings.

**Lineup** — four different free-tier providers, four 2025+ generation model families, **all ≥15 RPM**, all $0:

| Seat | Model | Provider | Free RPM | Free RPD/cap | Structured output |
|---|---|---|---|---|---|
| 1 | `gemini-3.1-flash-lite` | Google AI Studio | **15** (user floor) | **1,500 RPD** | native `response_schema=<pydantic>` |
| 2 | `meta/llama-4-maverick-17b-128e-instruct` (or `nvidia/nemotron-3-super-120b-a12b`) | NVIDIA NIM (build.nvidia.com) | **40** | **no documented daily cap** ([costbench](https://costbench.com/software/llm-api-providers/nvidia-nim/free-plan/)) | OpenAI-compatible `response_format` |
| 3 | `mistral-small-latest` (or `magistral-latest`) | Mistral La Plateforme (Experiment tier) | **~60** (1 RPS) | **~1B tokens/month** (effectively unlimited at our scale) | first-class `response_format={"type":"json_schema","strict":true}` + `client.chat.parse()` Pydantic — strictest in lineup |
| 4 | `deepseek/deepseek-chat-v3.1:free` (or `z-ai/glm-4.6:free`) | OpenRouter free routes | **20** | **1,000 RPD** (requires one-time $10 top-up; credits never spent on `:free`) | OpenAI-compatible `response_format` |

User constraints honored: every seat ≥15 RPM; Gemini throttle floor is its actual free-tier ceiling (15 RPM).

All four are 2025+ generation across distinct families: Google (Gemini), Meta or NVIDIA-tuned (NIM), Mistral, and DeepSeek/Zhipu (OpenRouter). Each speaks an OpenAI-compatible API except Gemini (already wired) and Mistral (own `mistralai` SDK).

**Sources**:
- NVIDIA NIM rate limits: [yangmao.ai NVIDIA Build](https://yangmao.ai/en/providers/nvidia-build/), [costbench NVIDIA NIM free plan](https://costbench.com/software/llm-api-providers/nvidia-nim/free-plan/)
- Mistral pricing + structured output: [mistral.ai/pricing](https://mistral.ai/pricing), [Mistral structured output docs](https://docs.mistral.ai/capabilities/structured_output/)
- OpenRouter free-route rate limits: [OpenRouter docs](https://openrouter.zendesk.com/hc/en-us/articles/39501163636379), [datastudios.org OpenRouter limits](https://www.datastudios.org/post/openrouter-api-key-free-limits-free-routes-paid-access-and-byok)
- Gemini free tier: [Google AI Studio rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)

**Cost**:

| Component | Scope B | Scope A |
|---|---:|---:|
| Gemini free | $0 | $0 |
| NVIDIA NIM free | $0 | $0 |
| Mistral Experiment free | $0 | $0 |
| OpenRouter (`:free` routes) | $0 (one-time $10 top-up — credits not spent on free routes) | $0 |
| **Total** | **~$0** (or $10 one-time) | **~$0** (or $10 one-time) |

**Wall-clock** — RPD ceilings are the bottleneck, not RPM:

**Scope B (~2.16K calls per seat)**:

| Seat | RPM | RPD | If RPD-unbounded | Realistic |
|---|---:|---:|---|---|
| Gemini | 15 | **1,500** | 144 min | **2.16K > 1500 → 660 calls spill to day 2** |
| NVIDIA NIM | 40 | (none) | 54 min | ~1 h |
| Mistral | ~60 | ~1B tok/mo | 36 min | ~0.6 h |
| OpenRouter `:free` | 20 | **1,000** | 108 min | **2.16K > 1000 → 1160 calls spill to day 2** |

→ **Realistic Scope B wall-clock**: **~1.5 days** to land Gemini + OpenRouter's spill. NVIDIA + Mistral finish on day 1 (~1 h). The runner can write tier-6B-ready output for the fast seats day 1 and complete the slow seats overnight.

**Scope A (~8K calls per seat)**:

| Seat | RPD | Days needed |
|---|---:|---|
| Gemini | 1,500 | 8K / 1500 = **5.4 days** |
| NVIDIA NIM | none | 8K / 40 RPM = **3.3 h** |
| Mistral | ~1B tok/mo | **2.2 h** |
| OpenRouter | 1,000 | 8K / 1000 = **8 days** |

→ **Realistic Scope A wall-clock**: **~8 days** (OpenRouter-bound). If we drop OpenRouter (3-seat ensemble), Gemini still caps at 5.4 days. To compress further: substitute paid Gemini ($4 on Scope A) or paid OpenRouter, or fall back to Path 2.

**Optional escape hatch — EC2 fifth seat or RPD-bound seat replacement**:

If a Path 3 run is RPD-bound and the operator wants to escape without paying for any provider:

- Spin up `g6e.xlarge` (L40S 48GB, ~$1.86/h) running vLLM with **Qwen 3-32B 4-bit** or **Llama 4 Maverick fp8**. vLLM continuous-batching delivers ~100+ req/s on a 32B 4-bit, so 2.16K (Scope B) clears in ~30 min and 8K (Scope A) clears in ~2 h.
- Cost: ~**$1–4** (Scope B + setup) / ~**$5–10** (Scope A) — cheaper than the cheapest paid seat.
- Adds an `EC2VLLMClient` (OpenAI-compatible HTTP) + `scripts/deploy_ec2_vllm.{sh,ps1}` + `scripts/teardown_ec2_vllm.{sh,ps1}`.
- Region: `ap-southeast-1` (same as planned M5); `Project=daccord` tag; teardown committed before stand-up.

**Code surface** (new beyond Path 2):
- New clients in [src/daccord/eval/clients.py](../src/daccord/eval/clients.py):
  - `NvidiaNimClient` (OpenAI-compatible, base_url `https://integrate.api.nvidia.com/v1`, `nvapi-` key)
  - `MistralClient` (native `mistralai` SDK — strictest JSON-schema, worth the separate client over going OpenAI-compatible)
  - `OpenRouterClient` (OpenAI-compatible, base_url `https://openrouter.ai/api/v1`)
  - Optional `EC2VLLMClient` (OpenAI-compatible, base_url = the EC2 endpoint URL)
- `Provider` Literal additions in [src/daccord/costs/config.py](../src/daccord/costs/config.py): `nvidia_nim`, `mistral`, `openrouter`, `ec2_vllm`. None of these have per-call pricing entries — they're free-tier (RPD-capped) or one-shot wall-clock cost.
- `[caps_requests_per_day]` entries for the new free-tier providers in [costs/config.toml](../costs/config.toml):
  ```toml
  nvidia_nim = 100000      # NIM publishes no daily cap; arbitrary high ceiling for tracker sanity
  mistral = 100000         # ~1B tok/month >> our workload; high ceiling for tracker sanity
  openrouter = 1000        # documented :free-route cap with $10 top-up
  ```
- The optional EC2 path adds `scripts/deploy_ec2_vllm.{sh,ps1}` + `scripts/teardown_ec2_vllm.{sh,ps1}`.

---

## Path 4 — Paid API ensemble hosted on cloud (laptop-decoupled)

**Status**: design pattern; thin shell-script wrapper around Path 2 to take it off the laptop.

**Why this exists**: Path 2's wall-clock at Scope A is **~9 h** (Anthropic 3.3 s/call + Gemini 4 s/call + Qwen 4 s/call + GPT-5-mini 4-5 s/call with `reasoning_effort="minimal"`). That's tolerable but inappropriate to run on a laptop — sleep, network drops, accidental Ctrl-C all cost the operator. **Path 4 = run Path 2 on a cheap cloud VM so the laptop can close.**

**Mechanism**:
- Provision a small EC2 instance (`t3.small` ≈ $0.02/h, or `t3.medium` ≈ $0.04/h for headroom).
- Push API keys (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_API_KEY` / `TOGETHER_API_KEY`) via env or SSM Parameter Store; **never bake into the AMI**.
- `git clone` the repo + run `docker compose run -e DACCORD_RPM_GOOGLE_GEMINI=4000 --rm root uv run python scripts/run_ensemble.py run-paid`.
- Sync `data/ensemble/raw/` back to the local repo when the run finishes (`aws s3 sync` to a Project=daccord-tagged S3 bucket, or `rsync` over SSH).
- Tear down the EC2 instance.

**OpenAI reasoning-effort fix bundled** (lands at the same time as Path 4): `OpenAIClient` now sets `reasoning_effort="minimal"` for GPT-5-family models. Without this, GPT-5-mini calls eat 64-500+ invisible reasoning tokens before emitting the visible 150-token answer (observed ~15 s/call in the live 72-pair attempt 2026-05-27). With it, per-call latency lands ~3-5 s, in line with the other Path-2 seats — so Path 2's per-pair wall-clock matches its sequential-throughput projection.

**Cost**:

| Component | Scope B (~$3) | Scope A (~$28) |
|---|---:|---:|
| API spend (same as Path 2) | ~$6.56 | ~$28 |
| EC2 t3.medium @ $0.04/h × wall-clock | ~$0.04 (1 h) | ~$0.36 (9 h) |
| S3 transfer-out + storage (data/ensemble/raw/ ~5 MB) | <$0.01 | <$0.01 |
| **Total** | **~$6.60** | **~$28.36** |

**Wall-clock** (with `reasoning_effort="minimal"` + sequential-per-seat):

- All four seats latency-bound (~3-5 s/call); bottleneck = slowest seat per pair.
- Scope B: ~1 h
- Scope A: **~9 h**

The 9-hour run goes on a VM that's a $4-cent-per-hour t3.medium with no operator attention. Resumability still applies — if the EC2 instance gets preempted (spot), restarting from `data/ensemble/raw/` picks up cleanly. For spot instances on the same workload, EC2 cost drops to ~$0.12 for the full run.

**EC2 region**: `ap-southeast-1` (same as the existing M2/M5 plan) keeps the `caravan-poc` profile + Project=daccord tag policies unchanged. RTT from `ap-southeast-1` to the four API providers' nearest regions averages 80-200 ms, materially the same as from the laptop. **Path 4 is NOT a network-latency win** — call latency is dominated by provider-side compute, which is identical whether you're calling from a laptop or an EC2 box.

**What Path 4 is NOT solving**:
- Not parallelism-within-seat. If you want to push GPT-5-mini's 500 RPM Tier-1 ceiling instead of its ~12 RPM sequential ceiling, that's a separate code change (replace `for prompt in remaining` with `asyncio.gather` or `ThreadPoolExecutor(max_workers=N)` per seat). EC2 doesn't change that.
- Not a different ensemble lineup. Same four 2025+ seats as Path 2.
- Not Path 3's free-tier story. Path 4 is paid-API run + cloud host.

**Code surface** (this MR adds the reasoning-effort fix; the EC2 provisioning script is a thin shell wrapper deferred to actual M2-run execution):
- ✓ `OpenAIClient` in [src/daccord/eval/clients.py](../src/daccord/eval/clients.py): `reasoning_effort="minimal"` for GPT-5 family.
- (deferred) `scripts/run_ec2_ensemble.sh` — boto3-shelled provisioner: launch t3.medium with Project=daccord tag, run setup commands via SSM (or SSH), block on the run, sync output to S3, terminate. Cleanest pattern: model after [scripts/aws_setup.sh](../scripts/aws_setup.sh).
- (deferred) Operator runbook updates in this doc once the provisioning script lands.

**When to pick Path 4 over Path 2**:
- Run > 2 hours expected (Scope A, or operator wants several Scope-B retries) → Path 4.
- One-off Scope B smoke or single-pair retry → Path 2 (laptop) is fine.
- Network/laptop reliability is a concern (long flight, hotel wifi, etc.) → Path 4.

---

## Path comparison

| Path | $ Scope B | Wall-clock B | $ Scope A | Wall-clock A | Status |
|---|---:|---:|---:|---:|---|
| **1 — Bedrock batch** | ~$8–12 | 1–24 h | ~$30–45 | 1–24 h | blocked |
| **2 — Paid direct API (local)** | **~$6.56** | **~1 h** | **~$28** | **~9 h** | laptop-bound; OK for Scope B + smokes; **2026-05-27 actual latency** = much higher than original "45 min Scope B" estimate |
| **3 — 4-provider free** | **~$0** (or $10 once) | **~1.5 days** (RPD-bound) | **~$0** (or $10 once) | **~8 days** (RPD-bound) | budget-zero option |
| **3 + EC2 escape** | ~$2–4 | ~30 min on the EC2 seat; Gemini + OpenRouter still RPD-bound | ~$5–10 | ~2 h on the EC2 seat | use when Path 3's RPD bottleneck on Gemini/OpenRouter is unacceptable |
| **4 — Path 2 on EC2** | **~$6.60** | **~1 h** (laptop-free) | **~$28.36** | **~9 h** (laptop-free) | **recommended for any run > 2 h**; laptop closed during run |

**Path 2 vs Path 4**: identical lineup + cost; Path 4 adds ~$0.36 EC2 fee + the operational benefit of "laptop closed for 9 hours." For Scope A the EC2 fee is a rounding error against the $28 API spend. **Recommend Path 4 for any run > 2 h.**

**Path 3** saves all of Path 2/4's ~$28 in exchange for ~8 days wall-clock and signing up for 3 new free-tier accounts (NVIDIA NIM, Mistral La Plateforme, OpenRouter). Useful when budget is truly zero; otherwise the time premium is too high.

**Path 1** is preserved (modularized as `BedrockBatchStrategy`) for when the AWS account unblocks.

---

## Shared infrastructure (used by all three paths)

### Hoist the prompt shape

Move `BatchPrompt` + `model_slug` from [src/daccord/aws/batch.py](../src/daccord/aws/batch.py) to new `src/daccord/ensemble/prompt.py`. Re-export from `daccord.aws.batch` so [tests/test_bedrock_batch.py](../tests/test_bedrock_batch.py) keeps passing.

### `EnsembleStrategy` protocol

New `src/daccord/ensemble/strategy.py`:

```python
class EnsembleStrategy(Protocol):
    name: str               # "bedrock-batch" | "bedrock-sync" | "local-api-paid" | "local-api-free-ec2"
    models: list[str]
    def run_pair(
        self,
        framework_pair: str,
        prompts: list[BatchPrompt],
        out_dir: Path,
        *,
        smoke: bool,
    ) -> dict[str, RunResult]: ...
```

`RunResult` (also new): `{model, output_path, parse_ok, parse_errors, seconds_elapsed}`.

### Per-provider throttle rewrite in [_rpm.py](../src/daccord/eval/_rpm.py)

Current: single 10-RPM global deque, not thread-safe. Replace with:

```python
_PROVIDER_RPM: dict[Provider, int] = {
    # Free tier (verified ceilings; user floor: ≥15)
    "google_gemini": 15,
    "groq": 28,
    "nvidia_nim": 35,
    "mistral": 50,
    "openrouter": 18,
    # Paid Tier 1 (verified via research, ~10% safety margin)
    "anthropic": 45,
    "openai": 450,
    "together": 600,
    "deepseek": 50,
    # Self-hosted
    "ec2_vllm": 600,
}
_CALL_TIMES: dict[Provider, deque[float]] = defaultdict(deque)
_LOCK = threading.Lock()

def api_throttle(provider: Provider) -> None: ...
```

Each `*Client.generate` passes `self.provider` to `api_throttle(provider)`. Adds a `threading.Lock` around deque mutation since `LocalAPIStrategy` uses a ThreadPoolExecutor.

### CLI wiring in [scripts/run_ensemble.py](../scripts/run_ensemble.py)

- `--strategy {bedrock-batch, bedrock-sync, local-api-paid, local-api-free-ec2}` (default `local-api-paid`).
- `--max-clauses-per-pair N` surfaced on submit + run subparsers.
- `--local-models` comma-list override (defaults per strategy).
- Strategy-specific args: `--profile` (Bedrock paths), `--ec2-endpoint` (free-EC2 path).
- Smoke mode works for all four strategies.

### Cost-config additions

Append to [costs/config.toml](../costs/config.toml):

```toml
[pricing.anthropic."claude-haiku-4-5"]
input_per_mtok = 1.00
output_per_mtok = 5.00

[pricing.openai."gpt-5-mini"]
input_per_mtok = 0.25
output_per_mtok = 2.00

[pricing.google_gemini."gemini-3.1-flash-lite"]
input_per_mtok = 0.25
output_per_mtok = 1.50

[pricing.together."Qwen/Qwen3-235B-A22B-Instruct-2507-tput"]
input_per_mtok = 0.20
output_per_mtok = 0.60
```

`[caps_usd_per_day]`:
- `anthropic` → **$25** (Scope A budget headroom: $14 with ~75% margin).
- `google_gemini` → add USD cap **$10** alongside existing RPD entry.
- `openai`, `together` existing entries cover the worst case.
- `ec2_vllm` USD cap **$20** (covers a full Scope A day on g6e.xlarge with margin).

Add `ec2_vllm` to `Provider` Literal in [src/daccord/costs/config.py](../src/daccord/costs/config.py).

---

## Bottom line

- Four execution paths behind `EnsembleStrategy`: Bedrock (preserved, blocked), paid direct API local (Path 2), 4-provider free (Path 3, budget-zero), paid direct API on EC2 (Path 4, **recommended for runs > 2 h**).
- **Path 2** (local) and **Path 4** (EC2) share the same lineup: Haiku 4.5 + GPT-5-mini + Gemini 3.1 Flash Lite + Qwen 3-235B-A22B Instruct via Together. **~$6.56 / ~1 h** for Scope B; **~$28 / ~9 h** for Scope A (with `reasoning_effort="minimal"` for GPT-5).
- **Path 3** (Gemini + NVIDIA NIM + Mistral + OpenRouter `:free`) — **$0 / ~1.5 days** for Scope B; **$0 / ~8 days** for Scope A. All seats ≥15 RPM; bottlenecked by Gemini (1500 RPD) + OpenRouter (1000 RPD).
- Tier-6B consumer of `data/ensemble/raw/*.jsonl` unchanged across all paths.
