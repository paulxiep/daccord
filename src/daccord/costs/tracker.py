from __future__ import annotations

import csv
import os
from datetime import UTC, datetime
from pathlib import Path

from daccord.costs.config import Provider, daily_csv_path, load_config
from daccord.costs.errors import CapExceeded
from daccord.costs.storage import CallRow, append_call, daily_rows, sum_today
from daccord.validation import validated

OVERRIDE_ENV = "DACCORD_COSTS_OVERRIDE"

DAILY_CSV_HEADER = (
    "date",
    "provider",
    "model",
    "input_tokens",
    "output_tokens",
    "n_calls",
    "cost_usd",
)


def _override_active() -> bool:
    return os.environ.get(OVERRIDE_ENV, "") == "1"


@validated
def estimate_cost(provider: Provider, model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = load_config().pricing_for(provider, model)
    return (
        input_tokens * pricing.input_per_mtok + output_tokens * pricing.output_per_mtok
    ) / 1_000_000.0


@validated
def today_spend(provider: Provider) -> float:
    return sum_today(provider)


@validated
def preflight(
    provider: Provider, model: str, est_input_tokens: int, est_output_tokens: int
) -> float:
    est_cost = estimate_cost(provider, model, est_input_tokens, est_output_tokens)
    if _override_active():
        return est_cost
    cap = load_config().cap_for(provider)
    spent = today_spend(provider)
    if spent + est_cost > cap:
        raise CapExceeded(
            f"{provider} preflight: today's spend ${spent:.4f} + est ${est_cost:.4f} "
            f"> cap ${cap:.2f}. Resume tomorrow or set {OVERRIDE_ENV}=1."
        )
    return est_cost


@validated
def record_call(
    provider: Provider,
    model: str,
    input_tokens: int,
    output_tokens: int,
    run_id: str | None = None,
    batch_id: str | None = None,
) -> float:
    cost = estimate_cost(provider, model, input_tokens, output_tokens)
    row = CallRow(
        ts_utc=datetime.now(UTC).isoformat(timespec="microseconds"),
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        run_id=run_id,
        batch_id=batch_id,
    )
    append_call(row)
    if not _override_active():
        cap = load_config().cap_for(provider)
        spent = today_spend(provider)
        if spent > cap:
            raise CapExceeded(
                f"{provider} record_call: today's spend ${spent:.4f} > cap ${cap:.2f} "
                f"after recording. Call already logged. "
                f"Resume tomorrow or set {OVERRIDE_ENV}=1."
            )
    return cost


@validated
def rollup_daily() -> Path:
    target = daily_csv_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = daily_rows()
    with target.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(DAILY_CSV_HEADER)
        writer.writerows(
            (r.date, r.provider, r.model, r.input_tokens, r.output_tokens, r.n_calls, f"{r.cost_usd:.4f}")
            for r in rows
        )
    return target
