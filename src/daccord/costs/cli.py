from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

from daccord.costs.config import PROVIDERS, Provider, load_config
from daccord.costs.storage import daily_provider_totals
from daccord.costs.tracker import rollup_daily, today_requests, today_spend


def _streak_days_over(
    per_day: dict[str, dict[Provider, float]], provider: Provider, threshold: float
) -> int:
    """Length of the most recent consecutive run of days where provider spent >= threshold,
    counting backwards from the most recent recorded date."""
    dates_desc = sorted(per_day.keys(), reverse=True)
    streak = 0
    for d in dates_desc:
        if per_day[d].get(provider, 0.0) >= threshold:
            streak += 1
        else:
            break
    return streak


def _flag(spent: float, cap: float, warn: float) -> str:
    if spent > cap:
        return "[OVER CAP]"
    if spent >= warn:
        return f"[WARN >=${warn:.0f}]"
    return "[OK]"


def _flag_rpd(n: int, cap: int) -> str:
    if n > cap:
        return "[OVER RPD]"
    if n >= int(cap * 0.8):
        return "[WARN >=80%]"
    return "[OK]"


def cmd_status(_args: argparse.Namespace) -> int:
    config = load_config()
    today = datetime.now(UTC).date().isoformat()
    per_day = daily_provider_totals()
    # Local-only providers (e.g. "retrieval") have no cap entry; `kind_of`
    # would raise on them. Check membership directly so PROVIDERS can include
    # local-only entries without breaking status reporting.
    paid_providers: list[Provider] = [p for p in PROVIDERS if p in config.caps_usd_per_day]
    free_providers: list[Provider] = [p for p in PROVIDERS if p in config.caps_requests_per_day]
    streaks = {
        p: _streak_days_over(per_day, p, config.warning_threshold_usd) for p in paid_providers
    }
    print(f"D'accord cost status  ({today} UTC)")
    print(
        f"  warning >= ${config.warning_threshold_usd:.2f}/d   "
        f"alert at {config.consecutive_days_for_alert}+ consecutive days\n"
    )
    for p in paid_providers:
        cap = config.cap_for(p)
        spent = today_spend(p)
        flag = _flag(spent, cap, config.warning_threshold_usd)
        print(f"  {p:<14}  today ${spent:>7.4f} / cap ${cap:>5.2f}   streak {streaks[p]}d  {flag}")
    for p in free_providers:
        rpd_cap = config.request_cap_for(p)
        n = today_requests(p)
        flag = _flag_rpd(n, rpd_cap)
        print(f"  {p:<14}  today {n:>5} req / cap {rpd_cap:>5} RPD   {flag}")
    alerts = [p for p, s in streaks.items() if s >= config.consecutive_days_for_alert]
    if alerts:
        print(f"\n  ALERT  R7 streak hit for: {', '.join(alerts)}", file=sys.stderr)
        return 1
    return 0


def cmd_rollup(_args: argparse.Namespace) -> int:
    path = rollup_daily()
    print(f"wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="daccord.costs", description="D'accord API spend tracker."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser(
        "status", help="print today's spend per provider; exit 1 if R7 alert"
    ).set_defaults(func=cmd_status)
    sub.add_parser("rollup", help="rebuild costs/daily.csv from inflight.sqlite").set_defaults(
        func=cmd_rollup
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
