# `costs/` — API spend tracking

Per-provider daily caps for the Phase-1 ensemble + eval API spend. See [docs/development_plan.md](../docs/development_plan.md) §6 and the R7 risk in §7 for the policy this implements.

## Files

| File | Committed? | Purpose |
|---|---|---|
| `config.toml` | **yes** | Caps + pricing table. Audit trail via git history. |
| `daily.csv` | **yes** | Rolled-up spend log: `date,provider,model,input_tokens,output_tokens,n_calls,cost_usd`. |
| `inflight.sqlite` | no | Per-call event log (WAL mode). Source for `daily.csv`. Regenerable. |

## Public API (for tiers 3A, 7A, eval)

```python
from daccord.costs import preflight, record_call, today_spend, rollup_daily
from daccord.costs import CapExceeded, UnknownModel

# Before each API call:
est = preflight("anthropic", "claude-3-5-sonnet-20241022",
                est_input_tokens=2000, est_output_tokens=500)
# ... make the API call ...
record_call("anthropic", "claude-3-5-sonnet-20241022",
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            run_id="ensemble-2026-05-23",
            batch_id="GDPR__PDPA-SG")

# Anytime:
spent = today_spend("anthropic")  # USD
rollup_daily()                    # rebuild daily.csv from inflight.sqlite
```

`preflight` raises `CapExceeded` if today's spend + estimated cost would exceed the per-provider daily cap. `record_call` raises defensively if a recorded call pushed past cap.

**Override** (knowingly push through a cap): `DACCORD_COSTS_OVERRIDE=1`.

## CLI

```bash
uv run python -m daccord.costs status   # exit 1 if R7 streak fires
uv run python -m daccord.costs rollup   # rebuild daily.csv
```

`status` flags each provider `[OK]` / `[WARN >=$25]` / `[OVER CAP]` and exits 1 if any provider has been ≥ `warning_threshold_usd` for ≥ `consecutive_days_for_alert` consecutive days (R7 early-warning signal).

## Adding a new model

Edit `config.toml`, append a table:

```toml
[pricing.openai."gpt-4o-2026-some-new-version"]
input_per_mtok = 2.50
output_per_mtok = 10.00
```

Commit. No code change required. Unknown `(provider, model)` at call time raises `UnknownModel` — there is no silent zero-cost fallback.

## Adjusting caps

Edit `caps_usd_per_day` in `config.toml`. Commit the change so the git log shows when the policy moved.
