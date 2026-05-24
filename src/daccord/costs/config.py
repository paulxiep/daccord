from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from daccord.validation import ValidatedModel, validated

Provider = Literal[
    "anthropic",
    "openai",
    "together",
    "groq",
    "google_gemini",
    "cerebras",
    "deepseek",
    # Local-only providers — never go through preflight/record_call, so they
    # need no cap entry in costs/config.toml. Added so ModelClient adapters
    # for local-inference baselines (retrieval index, future LocalHFClient
    # for base-Qwen) can satisfy the Protocol's `provider: Provider` field
    # without polluting the cost-tracking codepath.
    "retrieval",
]
PROVIDERS: tuple[Provider, ...] = (
    "anthropic",
    "openai",
    "together",
    "groq",
    "google_gemini",
    "cerebras",
    "deepseek",
    "retrieval",
)
ProviderKind = Literal["paid", "free_tier"]

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
    caps_usd_per_day: dict[Provider, float] = Field(default_factory=dict)
    caps_requests_per_day: dict[Provider, int] = Field(default_factory=dict)
    pricing: dict[Provider, dict[str, ModelPricing]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _provider_has_exactly_one_cap(self) -> Self:
        paid = set(self.caps_usd_per_day)
        free = set(self.caps_requests_per_day)
        overlap = paid & free
        if overlap:
            raise ValueError(
                f"providers in both caps_usd_per_day and caps_requests_per_day: {sorted(overlap)}"
            )
        return self

    def kind_of(self, provider: Provider) -> ProviderKind:
        if provider in self.caps_usd_per_day:
            return "paid"
        if provider in self.caps_requests_per_day:
            return "free_tier"
        raise KeyError(f"provider {provider!r} has no cap configured in costs/config.toml")

    def cap_for(self, provider: Provider) -> float:
        return self.caps_usd_per_day[provider]

    def request_cap_for(self, provider: Provider) -> int:
        return self.caps_requests_per_day[provider]

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
