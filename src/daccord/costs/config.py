from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from daccord.validation import ValidatedModel, validated

Provider = Literal["anthropic", "openai", "together"]
PROVIDERS: tuple[Provider, ...] = ("anthropic", "openai", "together")

REPO_ROOT_ENV = "DACCORD_REPO_ROOT"
CONFIG_PATH_ENV = "DACCORD_COSTS_CONFIG"
INFLIGHT_PATH_ENV = "DACCORD_COSTS_INFLIGHT"
DAILY_CSV_PATH_ENV = "DACCORD_COSTS_DAILY_CSV"


class ModelPricing(ValidatedModel):
    input_per_mtok: float
    output_per_mtok: float


class CostsConfig(ValidatedModel):
    warning_threshold_usd: float
    consecutive_days_for_alert: int
    caps_usd_per_day: dict[Provider, float]
    pricing: dict[Provider, dict[str, ModelPricing]]

    def cap_for(self, provider: Provider) -> float:
        return self.caps_usd_per_day[provider]

    def pricing_for(self, provider: Provider, model: str) -> ModelPricing:
        from daccord.costs.errors import UnknownModel

        provider_table = self.pricing.get(provider, {})
        if model not in provider_table:
            raise UnknownModel(f"no pricing for ({provider!r}, {model!r}) in costs/config.toml")
        return provider_table[model]


def _repo_root() -> Path:
    override = os.environ.get(REPO_ROOT_ENV)
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[3]


def config_path() -> Path:
    override = os.environ.get(CONFIG_PATH_ENV)
    return Path(override).resolve() if override else _repo_root() / "costs" / "config.toml"


def inflight_path() -> Path:
    override = os.environ.get(INFLIGHT_PATH_ENV)
    return Path(override).resolve() if override else _repo_root() / "costs" / "inflight.sqlite"


def daily_csv_path() -> Path:
    override = os.environ.get(DAILY_CSV_PATH_ENV)
    return Path(override).resolve() if override else _repo_root() / "costs" / "daily.csv"


@validated
def load_config(path: Path | None = None) -> CostsConfig:
    target = path if path is not None else config_path()
    raw = tomllib.loads(target.read_text(encoding="utf-8"))
    return CostsConfig.model_validate(raw)
