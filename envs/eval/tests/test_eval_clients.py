"""Tests for the eval-harness client adapters.

Network-free: both SDK clients are monkey-patched at the module level.
Validates two contracts:
  1. The adapter's JSON parsing maps SDK output to `CitationCandidate` and
     surfaces parse failures as `parse_error` instead of raising.
  2. Every successful `generate()` call routes through `costs.preflight` +
     `costs.record_call` with the correct provider+model+token attribution
     — the cap discipline that gates Phase-1 spend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daccord.costs.config import (
    CONFIG_PATH_ENV,
    DAILY_CSV_PATH_ENV,
    INFLIGHT_PATH_ENV,
)
from daccord.eval.clients import GeminiClient, GroqClient, _parse_candidate
from daccord.eval.schema import PromptMessages

COSTS_TOML = """\
warning_threshold_usd = 25.0
consecutive_days_for_alert = 2

[caps_usd_per_day]
anthropic = 30.0
openai = 20.0
together = 15.0

[caps_requests_per_day]
groq = 14400
google_gemini = 1500

[pricing.anthropic."claude-3-5-sonnet-20241022"]
input_per_mtok = 3.00
output_per_mtok = 15.00
"""


@pytest.fixture
def costs_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config_file = tmp_path / "config.toml"
    config_file.write_text(COSTS_TOML, encoding="utf-8")
    monkeypatch.setenv(CONFIG_PATH_ENV, str(config_file))
    monkeypatch.setenv(INFLIGHT_PATH_ENV, str(tmp_path / "inflight.sqlite"))
    monkeypatch.setenv(DAILY_CSV_PATH_ENV, str(tmp_path / "daily.csv"))
    monkeypatch.delenv("DACCORD_COSTS_OVERRIDE", raising=False)
    return tmp_path


MESSAGES = PromptMessages(system="sys", user="usr")
VALID_PAYLOAD = {
    "citation_id": "Section 24",
    "target_mechanism": "Reasonable security arrangements.",
    "mapping_justification": "Both require appropriate security.",
}


class _FakeUsage:
    def __init__(self, prompt: int, completion: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeGroqResponse:
    def __init__(self, content: str, usage: _FakeUsage | None) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = usage


class _FakeGroqCompletions:
    def __init__(self, content: str, usage: _FakeUsage | None) -> None:
        self._content = content
        self._usage = usage
        self.captured_kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> _FakeGroqResponse:
        self.captured_kwargs = kwargs
        return _FakeGroqResponse(self._content, self._usage)


class _FakeGroqChat:
    def __init__(self, completions: _FakeGroqCompletions) -> None:
        self.completions = completions


class _FakeGroqSDK:
    def __init__(self, content: str, usage: _FakeUsage | None) -> None:
        self.chat = _FakeGroqChat(_FakeGroqCompletions(content, usage))


def _make_groq_client(
    monkeypatch: pytest.MonkeyPatch, content: str, usage: _FakeUsage | None
) -> tuple[GroqClient, _FakeGroqCompletions]:
    fake_sdk = _FakeGroqSDK(content, usage)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr("groq.Groq", lambda **_kw: fake_sdk)
    client = GroqClient()
    return client, fake_sdk.chat.completions


class TestParseCandidate:
    def test_valid_payload_parses(self) -> None:
        candidate, err = _parse_candidate(json.dumps(VALID_PAYLOAD))
        assert err is None
        assert candidate is not None
        assert candidate.citation_id == "Section 24"

    def test_malformed_json_returns_error(self) -> None:
        candidate, err = _parse_candidate("not even close to json")
        assert candidate is None
        assert err is not None and "json decode" in err

    def test_missing_required_field_returns_error(self) -> None:
        bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "target_mechanism"}
        candidate, err = _parse_candidate(json.dumps(bad))
        assert candidate is None
        assert err is not None and "schema validation" in err

    def test_non_object_json_returns_error(self) -> None:
        candidate, err = _parse_candidate("[1, 2, 3]")
        assert candidate is None
        assert err is not None and "expected JSON object" in err


class TestGroqClient:
    def test_init_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        # Patch the SDK so the import inside __init__ succeeds even without keys
        monkeypatch.setattr("groq.Groq", lambda **_kw: None)
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            GroqClient()

    def test_generate_records_call_and_parses(
        self, costs_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, completions = _make_groq_client(
            monkeypatch, json.dumps(VALID_PAYLOAD), _FakeUsage(120, 80)
        )
        resp = client.generate(MESSAGES, run_id="run-x", batch_id="batch-y")

        # Schema mapping
        assert resp.top1 is not None
        assert resp.top1.citation_id == "Section 24"
        assert resp.parse_error is None
        assert resp.input_tokens == 120
        assert resp.output_tokens == 80
        assert resp.latency_ms >= 0

        # Cost-tracker integration: today_requests bumps by 1 for free-tier
        from daccord.costs import today_requests

        assert today_requests("groq") == 1

        # SDK call shape (regression guard)
        assert completions.captured_kwargs["model"] == "llama-3.3-70b-versatile"
        assert completions.captured_kwargs["response_format"] == {"type": "json_object"}
        assert completions.captured_kwargs["temperature"] == 0.0
        assert completions.captured_kwargs["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ]

    def test_generate_surfaces_parse_error(
        self, costs_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _ = _make_groq_client(monkeypatch, "not json", _FakeUsage(50, 5))
        resp = client.generate(MESSAGES, run_id="r", batch_id="b")
        assert resp.top1 is None
        assert resp.parse_error is not None
        # Call still recorded (cap accounting must see all traffic)
        from daccord.costs import today_requests

        assert today_requests("groq") == 1

    def test_generate_uses_estimate_when_usage_missing(
        self, costs_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _ = _make_groq_client(monkeypatch, json.dumps(VALID_PAYLOAD), usage=None)
        resp = client.generate(MESSAGES, run_id="r", batch_id="b")
        # est = (len("sys")+len("usr"))//4 = 6//4 = 1; output falls back to len(raw)//4
        assert resp.input_tokens == 1
        assert resp.output_tokens >= 1


class _FakeGeminiUsage:
    def __init__(self, prompt: int, completion: int) -> None:
        self.prompt_token_count = prompt
        self.candidates_token_count = completion


class _FakeGeminiResponse:
    def __init__(self, text: str, usage: _FakeGeminiUsage | None) -> None:
        self.text = text
        self.usage_metadata = usage


class _FakeGeminiModels:
    def __init__(self, text: str, usage: _FakeGeminiUsage | None) -> None:
        self._text = text
        self._usage = usage
        self.captured_kwargs: dict[str, Any] = {}

    def generate_content(self, **kwargs: Any) -> _FakeGeminiResponse:
        self.captured_kwargs = kwargs
        return _FakeGeminiResponse(self._text, self._usage)


class _FakeGeminiSDK:
    def __init__(self, text: str, usage: _FakeGeminiUsage | None) -> None:
        self.models = _FakeGeminiModels(text, usage)


def _make_gemini_client(
    monkeypatch: pytest.MonkeyPatch, text: str, usage: _FakeGeminiUsage | None
) -> tuple[GeminiClient, _FakeGeminiModels]:
    fake_sdk = _FakeGeminiSDK(text, usage)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr("google.genai.Client", lambda **_kw: fake_sdk)
    client = GeminiClient()
    return client, fake_sdk.models


class TestGeminiClient:
    def test_init_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setattr("google.genai.Client", lambda **_kw: None)
        with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
            GeminiClient()

    def test_generate_records_call_and_parses(
        self, costs_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, models = _make_gemini_client(
            monkeypatch, json.dumps(VALID_PAYLOAD), _FakeGeminiUsage(200, 100)
        )
        resp = client.generate(MESSAGES, run_id="run-x", batch_id="batch-y")

        assert resp.top1 is not None
        assert resp.top1.citation_id == "Section 24"
        assert resp.input_tokens == 200
        assert resp.output_tokens == 100

        from daccord.costs import today_requests

        assert today_requests("google_gemini") == 1

        # SDK call shape — json schema constraint must be wired
        assert models.captured_kwargs["model"] == "gemini-2.5-flash"
        cfg = models.captured_kwargs["config"]
        assert cfg.system_instruction == "sys"
        assert cfg.temperature == 0.0
        assert cfg.response_mime_type == "application/json"
        assert cfg.response_json_schema is not None
        assert "citation_id" in cfg.response_json_schema["properties"]
