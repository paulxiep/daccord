from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from daccord.costs import (
    CapExceeded,
    UnknownModel,
    estimate_cost,
    preflight,
    record_call,
    rollup_daily,
    today_requests,
    today_spend,
)
from daccord.costs import cli as cli_module
from daccord.costs.config import (
    CONFIG_PATH_ENV,
    DAILY_CSV_PATH_ENV,
    INFLIGHT_PATH_ENV,
    load_config,
)
from daccord.costs.storage import CallRow, append_call

CONFIG_TOML = """\
warning_threshold_usd = 25.0
consecutive_days_for_alert = 2

[caps_usd_per_day]
anthropic = 30.0
openai = 20.0
together = 15.0

[caps_requests_per_day]
groq = 14400
google_gemini = 1500
cerebras = 1000
deepseek = 1000

[pricing.anthropic."claude-3-5-sonnet-20241022"]
input_per_mtok = 3.00
output_per_mtok = 15.00

[pricing.openai."gpt-4o"]
input_per_mtok = 2.50
output_per_mtok = 10.00

[pricing.together."Qwen/Qwen2.5-72B-Instruct-Turbo"]
input_per_mtok = 1.20
output_per_mtok = 1.20
"""


@pytest.fixture
def costs_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config_file = tmp_path / "config.toml"
    config_file.write_text(CONFIG_TOML, encoding="utf-8")
    inflight = tmp_path / "inflight.sqlite"
    daily = tmp_path / "daily.csv"
    monkeypatch.setenv(CONFIG_PATH_ENV, str(config_file))
    monkeypatch.setenv(INFLIGHT_PATH_ENV, str(inflight))
    monkeypatch.setenv(DAILY_CSV_PATH_ENV, str(daily))
    monkeypatch.delenv("DACCORD_COSTS_OVERRIDE", raising=False)
    return tmp_path


class TestPricing:
    def test_estimate_cost_sonnet(self, costs_env: Path) -> None:
        # 1M in @ $3 + 1M out @ $15 = $18
        assert estimate_cost(
            "anthropic", "claude-3-5-sonnet-20241022", 1_000_000, 1_000_000
        ) == pytest.approx(18.0)

    def test_estimate_cost_gpt4o_partial(self, costs_env: Path) -> None:
        # 500k in @ $2.50/MTok + 100k out @ $10/MTok = 1.25 + 1.00 = 2.25
        assert estimate_cost("openai", "gpt-4o", 500_000, 100_000) == pytest.approx(2.25)

    def test_unknown_model_raises(self, costs_env: Path) -> None:
        with pytest.raises(UnknownModel):
            estimate_cost("openai", "gpt-4o-not-a-real-version", 100, 50)

    def test_unknown_provider_rejected_by_validation(self, costs_env: Path) -> None:
        with pytest.raises(ValidationError):
            estimate_cost("googleai", "gemini-pro", 100, 50)  # type: ignore[arg-type]


class TestCaps:
    def test_preflight_under_cap_passes(self, costs_env: Path) -> None:
        # cap $30; 100k+100k sonnet = (0.3 + 1.5) = $1.80
        assert preflight(
            "anthropic", "claude-3-5-sonnet-20241022", 100_000, 100_000
        ) == pytest.approx(1.80)

    def test_preflight_over_cap_raises(self, costs_env: Path) -> None:
        # 2M in + 2M out sonnet = (6 + 30) = $36 > $30 cap
        with pytest.raises(CapExceeded):
            preflight("anthropic", "claude-3-5-sonnet-20241022", 2_000_000, 2_000_000)

    def test_record_call_under_cap(self, costs_env: Path) -> None:
        cost = record_call("openai", "gpt-4o", 100_000, 100_000)  # 0.25 + 1.00 = 1.25
        assert cost == pytest.approx(1.25)
        assert today_spend("openai") == pytest.approx(1.25)

    def test_record_call_pushing_over_cap_raises(self, costs_env: Path) -> None:
        # together cap $15; charge a $20 call -> raise after record
        with pytest.raises(CapExceeded):
            record_call("together", "Qwen/Qwen2.5-72B-Instruct-Turbo", 10_000_000, 10_000_000)
        # call still logged (defensive raise after append)
        assert today_spend("together") > 15.0

    def test_override_bypasses_preflight(
        self, costs_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DACCORD_COSTS_OVERRIDE", "1")
        preflight("anthropic", "claude-3-5-sonnet-20241022", 2_000_000, 2_000_000)

    def test_override_bypasses_record_call(
        self, costs_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DACCORD_COSTS_OVERRIDE", "1")
        record_call("together", "Qwen/Qwen2.5-72B-Instruct-Turbo", 10_000_000, 10_000_000)


class TestConcurrency:
    def test_threaded_record_calls_conserve_totals(self, costs_env: Path) -> None:
        n_threads, n_calls = 8, 50
        # 1k in + 1k out sonnet = (0.003 + 0.015) = $0.018 per call
        per_call = estimate_cost("anthropic", "claude-3-5-sonnet-20241022", 1_000, 1_000)
        expected_total = n_threads * n_calls * per_call

        def worker() -> None:
            for _ in range(n_calls):
                record_call("anthropic", "claude-3-5-sonnet-20241022", 1_000, 1_000)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert today_spend("anthropic") == pytest.approx(expected_total, rel=1e-6)
        # Row-count sanity check via direct sqlite read
        with sqlite3.connect(str(costs_env / "inflight.sqlite")) as conn:
            (n,) = conn.execute("SELECT COUNT(*) FROM inflight").fetchone()
        assert n == n_threads * n_calls


class TestRollup:
    def test_rollup_groups_by_date_provider_model(self, costs_env: Path) -> None:
        record_call("anthropic", "claude-3-5-sonnet-20241022", 1_000_000, 0)
        record_call("anthropic", "claude-3-5-sonnet-20241022", 0, 100_000)
        record_call("openai", "gpt-4o", 100_000, 100_000)

        path = rollup_daily()
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "date,provider,model,input_tokens,output_tokens,n_calls,cost_usd"
        # 2 distinct (provider, model) groups for today
        assert len(lines) == 1 + 2

        # Re-running is idempotent
        path2 = rollup_daily()
        assert path2.read_text(encoding="utf-8") == path.read_text(encoding="utf-8")

    def test_rollup_aggregates_token_counts(self, costs_env: Path) -> None:
        for _ in range(3):
            record_call("openai", "gpt-4o", 1_000, 500)
        rollup_daily()
        rows = (costs_env / "daily.csv").read_text(encoding="utf-8").splitlines()
        assert len(rows) == 2  # header + 1 group
        date_str, provider, model, ins, outs, n_calls, cost = rows[1].split(",")
        assert provider == "openai"
        assert model == "gpt-4o"
        assert int(ins) == 3_000
        assert int(outs) == 1_500
        assert int(n_calls) == 3


class TestCliStatus:
    def _seed(self, dates: list[str], provider: str, model: str, cost_each: float) -> None:
        # Insert one synthetic high-cost row per date for the given provider.
        # Trip the cap check off by setting matching token totals so estimate matches.
        # Direct append_call writes whatever ts_utc/cost we provide.
        for d in dates:
            append_call(
                CallRow(
                    ts_utc=f"{d}T12:00:00+00:00",
                    provider=provider,  # type: ignore[arg-type]
                    model=model,
                    input_tokens=1,
                    output_tokens=1,
                    cost_usd=cost_each,
                )
            )

    def test_status_two_day_streak_exits_1(
        self, costs_env: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        today = datetime.now(UTC).date()
        d1 = today.replace(day=max(1, today.day - 1)).isoformat()
        d2 = today.isoformat()
        self._seed([d1, d2], "anthropic", "claude-3-5-sonnet-20241022", cost_each=26.0)

        rc = cli_module.main(["status"])
        captured = capsys.readouterr()
        assert rc == 1
        assert "ALERT" in captured.err
        assert "anthropic" in captured.err

    def test_status_single_day_under_threshold_exits_0(
        self, costs_env: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        today = datetime.now(UTC).date().isoformat()
        self._seed([today], "openai", "gpt-4o", cost_each=5.0)
        rc = cli_module.main(["status"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "[OK]" in captured.out


class TestConfigLoad:
    def test_load_config_returns_validated_model(self, costs_env: Path) -> None:
        cfg = load_config()
        assert cfg.warning_threshold_usd == 25.0
        assert cfg.cap_for("anthropic") == 30.0
        assert cfg.pricing_for("openai", "gpt-4o").input_per_mtok == 2.50

    def test_kind_of_paid(self, costs_env: Path) -> None:
        assert load_config().kind_of("anthropic") == "paid"

    def test_kind_of_free_tier(self, costs_env: Path) -> None:
        cfg = load_config()
        assert cfg.kind_of("groq") == "free_tier"
        assert cfg.kind_of("google_gemini") == "free_tier"
        assert cfg.kind_of("cerebras") == "free_tier"
        assert cfg.kind_of("deepseek") == "free_tier"

    def test_request_cap_for_free_tier(self, costs_env: Path) -> None:
        cfg = load_config()
        assert cfg.request_cap_for("groq") == 14400
        assert cfg.request_cap_for("google_gemini") == 1500
        assert cfg.request_cap_for("cerebras") == 1000
        assert cfg.request_cap_for("deepseek") == 1000


class TestFreeTier:
    def test_estimate_cost_is_zero_for_free_tier(self, costs_env: Path) -> None:
        assert (
            estimate_cost("groq", "meta-llama/llama-4-scout-17b-16e-instruct", 5_000, 5_000) == 0.0
        )
        assert estimate_cost("google_gemini", "gemini-3.1-flash-lite", 5_000, 5_000) == 0.0

    def test_preflight_under_rpd_cap_passes(self, costs_env: Path) -> None:
        # cap 14400; no calls today; preflight returns 0.0 cost
        assert preflight("groq", "meta-llama/llama-4-scout-17b-16e-instruct", 5_000, 5_000) == 0.0

    def test_preflight_over_rpd_cap_raises(self, costs_env: Path) -> None:
        # Seed 1500 calls under google_gemini today; preflight (which would push to 1501) raises
        today = datetime.now(UTC).date().isoformat()
        for i in range(1500):
            append_call(
                CallRow(
                    ts_utc=f"{today}T12:00:00.{i:06d}+00:00",
                    provider="google_gemini",
                    model="gemini-3.1-flash-lite",
                    input_tokens=1,
                    output_tokens=1,
                    cost_usd=0.0,
                )
            )
        with pytest.raises(CapExceeded):
            preflight("google_gemini", "gemini-3.1-flash-lite", 100, 100)

    def test_record_call_increments_request_count(self, costs_env: Path) -> None:
        assert today_requests("groq") == 0
        record_call("groq", "meta-llama/llama-4-scout-17b-16e-instruct", 1_000, 1_000)
        record_call("groq", "meta-llama/llama-4-scout-17b-16e-instruct", 1_000, 1_000)
        assert today_requests("groq") == 2
        # Free-tier rows carry $0 cost
        assert today_spend("groq") == 0.0

    def test_record_call_over_rpd_cap_raises(self, costs_env: Path) -> None:
        # Seed 1500 google_gemini calls then one more record should raise after append
        today = datetime.now(UTC).date().isoformat()
        for i in range(1500):
            append_call(
                CallRow(
                    ts_utc=f"{today}T12:00:00.{i:06d}+00:00",
                    provider="google_gemini",
                    model="gemini-3.1-flash-lite",
                    input_tokens=1,
                    output_tokens=1,
                    cost_usd=0.0,
                )
            )
        with pytest.raises(CapExceeded):
            record_call("google_gemini", "gemini-3.1-flash-lite", 100, 100)
        # Call was still logged before the raise
        assert today_requests("google_gemini") == 1501

    @pytest.mark.parametrize(
        ("provider", "model", "cap"),
        [
            ("cerebras", "qwen-3-235b-a22b-instruct-2507", 1000),
            ("deepseek", "deepseek-chat", 1000),
        ],
    )
    def test_preflight_over_rpd_cap_raises_new_providers(
        self, costs_env: Path, provider: str, model: str, cap: int
    ) -> None:
        today = datetime.now(UTC).date().isoformat()
        for i in range(cap):
            append_call(
                CallRow(
                    ts_utc=f"{today}T12:00:00.{i:06d}+00:00",
                    provider=provider,  # type: ignore[arg-type]
                    model=model,
                    input_tokens=1,
                    output_tokens=1,
                    cost_usd=0.0,
                )
            )
        with pytest.raises(CapExceeded):
            preflight(provider, model, 100, 100)  # type: ignore[arg-type]
        # Free-tier estimate_cost stays at zero for new providers
        assert estimate_cost(provider, model, 100, 100) == 0.0  # type: ignore[arg-type]

    def test_override_bypasses_rpd_preflight(
        self, costs_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        today = datetime.now(UTC).date().isoformat()
        for i in range(1500):
            append_call(
                CallRow(
                    ts_utc=f"{today}T12:00:00.{i:06d}+00:00",
                    provider="google_gemini",
                    model="gemini-3.1-flash-lite",
                    input_tokens=1,
                    output_tokens=1,
                    cost_usd=0.0,
                )
            )
        monkeypatch.setenv("DACCORD_COSTS_OVERRIDE", "1")
        # No raise even though we're past RPD cap
        preflight("google_gemini", "gemini-3.1-flash-lite", 100, 100)


class TestConfigValidation:
    def test_provider_in_both_caps_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bad = """\
warning_threshold_usd = 25.0
consecutive_days_for_alert = 2

[caps_usd_per_day]
groq = 5.0

[caps_requests_per_day]
groq = 14400
"""
        path = tmp_path / "bad.toml"
        path.write_text(bad, encoding="utf-8")
        monkeypatch.setenv(CONFIG_PATH_ENV, str(path))
        with pytest.raises(ValidationError):
            load_config()
