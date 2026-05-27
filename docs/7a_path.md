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
| 3 | `gemini-3.1-flash-lite` | Google | $0.25 | $1.50 | **300** (verify in AI Studio dashboard) | **1000 ⚠** (inherited from gemini-2.5-flash-lite T1 floor; preview-tier; **verify in dashboard**) | 1M | Auto → T2 at $100 cumulative + 3 days |
| 4 | `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` via Together | Meta | $0.27 | $0.85 | dynamic (concurrency-based; no fixed RPM published) | none | dynamic | No tiers — auto-adjusts with sustained traffic |

**Rate-limit sources** (verified May 2026):
- Anthropic: [platform.claude.com/docs/en/api/rate-limits](https://platform.claude.com/docs/en/api/rate-limits)
- OpenAI: [GPT-5-mini model page](https://developers.openai.com/api/docs/models/gpt-5-mini), [community rate-limit thread](https://community.openai.com/t/increased-gpt-5-and-gpt-5-mini-rate-limits/1357840)
- Gemini: [ai.google.dev rate-limits](https://ai.google.dev/gemini-api/docs/rate-limits) (defers to AI Studio dashboard — numbers above are the gemini-2.5-flash-lite T1 floor, widely reported as carrying over)
- Together: [docs.together.ai rate-limits](https://docs.together.ai/docs/rate-limits)

**Gemini RPD risk**: paid Tier 1 may inherit the gemini-2.5-flash-lite **1000 RPD** ceiling. If so, Scope A (8K Gemini calls per seat) takes ~8 days and even Scope B (2.16K) spills to ~2.2 days. Mitigations baked into the implementation: (a) `--max-clauses-per-pair` already exists; setting it to 30 caps Gemini below the 1000 RPD floor for ~3-day Scope-B-on-Gemini; (b) the resilient runner can resume across days; (c) Anthropic's $40 auto-tier-up to 1000 RPM gets us off the Anthropic-50-RPM bottleneck after the first spend, accelerating subsequent runs; (d) operator confirms the real Tier-1 RPD via [aistudio.google.com/rate-limit](https://aistudio.google.com/rate-limit) before kicking off the full run.

**Cost**:

| Seat | Scope B | Scope A |
|---|---:|---:|
| Haiku 4.5 | $3.67 | $14.13 |
| GPT-5-mini | $1.32 | $5.07 |
| Gemini 3.1 Flash Lite | $1.05 | $4.05 |
| Llama 4 Maverick (Together) | $0.72 | $2.79 |
| **Total** | **~$7** | **~$26** |

**Wall-clock** (in parallel, one thread per provider; bottleneck = slowest seat):
- Bottleneck (RPM): Haiku 4.5 at 50 RPM Tier 1 ⇒ Scope B 43 min, Scope A 160 min.
- Bottleneck (RPD, if Gemini's 1000 RPD ceiling holds): Gemini at 1000 RPD ⇒ Scope A 8 days; Scope B 2.2 days.
- **Best case** (RPD doesn't bite): Scope B ~45 min, Scope A ~2.7 h.
- **Worst case** (Gemini 1000 RPD ceiling): Scope B ~2.2 days, Scope A ~8 days.

Operator-side compress options: confirm Gemini Tier-1 RPD in [AI Studio](https://aistudio.google.com/rate-limit); if 1000 RPD holds, pre-spend $100 (one-shot) to auto-promote Gemini to Tier 2, OR cap `--max-clauses-per-pair 30` to land Scope B Gemini under 1000.

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

## Path comparison

| Path | $ Scope B | Wall-clock B | $ Scope A | Wall-clock A | Status |
|---|---:|---:|---:|---:|---|
| **1 — Bedrock batch** | ~$8–12 | 1–24 h | ~$30–45 | 1–24 h | blocked |
| **2 — Paid direct API** | **~$7** | **~45 min** | **~$26** | **~2.7 h** | recommended first pass |
| **3 — 4-provider free** | **~$0** (or $10 once) | **~1.5 days** (RPD-bound) | **~$0** (or $10 once) | **~8 days** (RPD-bound) | budget-zero option |
| **3 + EC2 escape** | ~$2–4 | ~30 min on the EC2 seat; Gemini + OpenRouter still RPD-bound | ~$5–10 | ~2 h on the EC2 seat | use when Path 3's RPD bottleneck on Gemini/OpenRouter is unacceptable |

**Path 2 dominates Path 1** on cost at the recommended scope (~$7 vs ~$8–12) AND is unblocked. **Path 3 saves all of Path 2's ~$7** in exchange for ~1.5 days wall-clock and signing up for 3 new free-tier accounts (NVIDIA NIM, Mistral La Plateforme, OpenRouter). If the operator has the patience and zero budget, Path 3 is fully viable. Path 2's 45-minute total + $7 is hard to argue against for the first pass.

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

[pricing.together."meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"]
input_per_mtok = 0.27
output_per_mtok = 0.85
```

`[caps_usd_per_day]`:
- `anthropic` → **$25** (Scope A budget headroom: $14 with ~75% margin).
- `google_gemini` → add USD cap **$10** alongside existing RPD entry.
- `openai`, `together` existing entries cover the worst case.
- `ec2_vllm` USD cap **$20** (covers a full Scope A day on g6e.xlarge with margin).

Add `ec2_vllm` to `Provider` Literal in [src/daccord/costs/config.py](../src/daccord/costs/config.py).

---

## Bottom line

- Three `EnsembleStrategy` impls — Bedrock (preserved), paid direct API (Path 2, recommended), 4-provider free API ensemble (Path 3, budget-zero with optional EC2 escape).
- **Path 2** (Haiku 4.5 + GPT-5-mini + Gemini 3.1 Flash Lite + Llama 4 Maverick) — **~$7 / ~45 min** for Scope B; **~$26 / ~2.7 h** for Scope A.
- **Path 3** (Gemini 3.1 Flash Lite + NVIDIA NIM + Mistral Experiment + OpenRouter `:free`) — **$0 / ~1.5 days** for Scope B; **$0 / ~8 days** for Scope A. All seats ≥15 RPM; bottlenecked by Gemini (1500 RPD) + OpenRouter (1000 RPD).
- Tier-6B consumer of `data/ensemble/raw/*.jsonl` unchanged across all three paths.
