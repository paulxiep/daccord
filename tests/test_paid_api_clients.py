"""Mocked tests for AnthropicClient / OpenAIClient / TogetherClient.

Each client is exercised by injecting a fake SDK module into `sys.modules`
before the client lazily imports its provider SDK. Covers:

  - Happy path: SDK returns the structured candidate, ModelResponse has
    `top1` populated, no parse_error.
  - SDK exception: client catches and surfaces as parse_error on the
    `ModelResponse` (so the strategy's per-call append still works).
  - Cost-tracker integration: preflight + record_call are exercised
    transparently via the existing `daccord.costs` plumbing (we test
    they don't crash, not their semantics — `tests/test_costs_*.py`
    own the semantic tests).

The eval-runner-level GroqClient / GeminiClient already have parallel
test coverage in envs/eval/tests/; these tests focus on the new tier-7A
clients only.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest

# Mark the whole module: each test injects fakes into sys.modules; we want
# them isolated per-test so cross-test pollution doesn't sneak through.


# ─────────────────────────────────────────────────────────────────────────────
# Shared fake-module helpers.
# ─────────────────────────────────────────────────────────────────────────────


def _set_env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set all four provider API keys so each client's __init__ doesn't raise."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("TOGETHER_API_KEY", "test-together")
    monkeypatch.setenv("DACCORD_COSTS_OVERRIDE", "1")  # bypass preflight caps


def _bypass_throttle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub `api_throttle` to no-op so tests don't wait on rate limiter."""
    import daccord.eval._rpm as _rpm

    monkeypatch.setattr(_rpm, "api_throttle", lambda _provider=None: None)
    # Also stub the alias inside clients module that imported it at load time.
    import daccord.eval.clients as _clients

    monkeypatch.setattr(_clients, "api_throttle", lambda _provider=None: None)


# ─────────────────────────────────────────────────────────────────────────────
# AnthropicClient — happy path + API error path.
# ─────────────────────────────────────────────────────────────────────────────


def _make_fake_anthropic_module(
    *, tool_input: dict[str, str] | None, raise_exc: Exception | None = None
) -> types.ModuleType:
    """Build a fake `anthropic` SDK with the bits AnthropicClient touches."""

    class FakeAPIError(Exception):
        pass

    class FakeAPITimeoutError(Exception):
        pass

    class FakeUsage:
        input_tokens = 100
        output_tokens = 50

    class FakeContentBlock:
        def __init__(self, kind: str, input_dict: dict[str, str] | None = None) -> None:
            self.type = kind
            if kind == "tool_use":
                self.input = input_dict or {}
            elif kind == "text":
                self.text = json.dumps(input_dict or {})

    class FakeResponse:
        def __init__(self, blocks: list[FakeContentBlock]) -> None:
            self.content = blocks
            self.usage = FakeUsage()

    class FakeMessages:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def create(self, **kwargs: Any) -> FakeResponse:
            self.calls.append(kwargs)
            if raise_exc is not None:
                raise raise_exc
            if tool_input is None:
                # Return a text block only (no tool_use) — simulates the
                # forced-tool-choice escape hatch we record as parse_error.
                return FakeResponse([FakeContentBlock("text", {"oops": "no tool"})])
            return FakeResponse([FakeContentBlock("tool_use", tool_input)])

    class FakeClient:
        def __init__(self, *, api_key: str, max_retries: int = 2) -> None:
            assert api_key  # __init__ contract
            self.max_retries = max_retries
            self.messages = FakeMessages()

    module = types.ModuleType("anthropic")
    module.Anthropic = FakeClient  # type: ignore[attr-defined]
    module.APIError = FakeAPIError  # type: ignore[attr-defined]
    module.APITimeoutError = FakeAPITimeoutError  # type: ignore[attr-defined]
    return module


def test_anthropic_client_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env_keys(monkeypatch)
    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        _make_fake_anthropic_module(
            tool_input={
                "citation_id": "PDPA-SG Sec 13",
                "target_mechanism": "Consent withdrawal mechanism",
                "mapping_justification": "Both require explicit consent",
            }
        ),
    )
    _bypass_throttle(monkeypatch)

    from daccord.eval.clients import AnthropicClient
    from daccord.eval.schema import PromptMessages

    client = AnthropicClient(model="claude-haiku-4-5")
    resp = client.generate(
        PromptMessages(system="sys", user="usr"),
        run_id="rid",
        batch_id="bid",
    )
    assert resp.parse_error is None
    assert resp.top1 is not None
    assert resp.top1.citation_id == "PDPA-SG Sec 13"
    assert resp.input_tokens == 100
    assert resp.output_tokens == 50


def test_anthropic_client_api_error_surfaces_as_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_env_keys(monkeypatch)
    # Build the fake module FIRST so we can pull its APIError class out and
    # use it as the raised exception inside the module itself (same class
    # identity end-to-end so the client's `except APIError` matches).
    fake_module = _make_fake_anthropic_module(tool_input=None)
    rate_limit_exc = fake_module.APIError("rate limit")  # type: ignore[attr-defined]
    fake_module_raising = _make_fake_anthropic_module(tool_input=None, raise_exc=rate_limit_exc)
    # Patch APIError on the raising-module so its raised instance is type-
    # compatible with the client's imported `APIError`.
    fake_module_raising.APIError = fake_module.APIError  # type: ignore[attr-defined]
    fake_module_raising.APITimeoutError = fake_module.APITimeoutError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake_module_raising)
    _bypass_throttle(monkeypatch)

    from daccord.eval.clients import AnthropicClient
    from daccord.eval.schema import PromptMessages

    client = AnthropicClient(model="claude-haiku-4-5")
    resp = client.generate(
        PromptMessages(system="sys", user="usr"),
        run_id="rid",
        batch_id="bid",
    )
    assert resp.top1 is None
    assert resp.parse_error is not None
    assert "rate limit" in resp.parse_error


def test_anthropic_client_no_tool_use_block_records_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_env_keys(monkeypatch)
    monkeypatch.setitem(sys.modules, "anthropic", _make_fake_anthropic_module(tool_input=None))
    _bypass_throttle(monkeypatch)

    from daccord.eval.clients import AnthropicClient
    from daccord.eval.schema import PromptMessages

    client = AnthropicClient(model="claude-haiku-4-5")
    resp = client.generate(PromptMessages(system="sys", user="usr"), run_id="r", batch_id="b")
    assert resp.top1 is None
    assert resp.parse_error is not None
    assert "tool_use" in resp.parse_error


# ─────────────────────────────────────────────────────────────────────────────
# OpenAIClient + TogetherClient — share the `openai` SDK; one fake module
# covers both. Tests assert per-client provider / model / cost-tag plumbing.
# ─────────────────────────────────────────────────────────────────────────────


_DEFAULT_OPENAI_PAYLOAD = (
    '{"citation_id": "x", "target_mechanism": "y", "mapping_justification": "z"}'
)


def _make_fake_openai_module(
    *,
    payload_json: str = _DEFAULT_OPENAI_PAYLOAD,
    raise_exc: Exception | None = None,
) -> types.ModuleType:
    """Build a fake `openai` SDK with chat.completions.create() returning JSON."""

    class FakeAPIError(Exception):
        pass

    class FakeAPITimeoutError(Exception):
        pass

    class FakeUsage:
        prompt_tokens = 80
        completion_tokens = 40

    class FakeMessage:
        def __init__(self, content: str) -> None:
            self.content = content

    class FakeChoice:
        def __init__(self, content: str) -> None:
            self.message = FakeMessage(content)

    class FakeResponse:
        def __init__(self, content: str) -> None:
            self.choices = [FakeChoice(content)]
            self.usage = FakeUsage()

    class FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def create(self, **kwargs: Any) -> FakeResponse:
            self.calls.append(kwargs)
            if raise_exc is not None:
                raise raise_exc
            return FakeResponse(payload_json)

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(
            self,
            *,
            api_key: str,
            base_url: str | None = None,
            max_retries: int = 2,
        ) -> None:
            assert api_key
            self.base_url = base_url
            self.max_retries = max_retries
            self.chat = FakeChat()

    module = types.ModuleType("openai")
    module.OpenAI = FakeClient  # type: ignore[attr-defined]
    module.APIError = FakeAPIError  # type: ignore[attr-defined]
    module.APITimeoutError = FakeAPITimeoutError  # type: ignore[attr-defined]
    return module


def test_openai_client_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env_keys(monkeypatch)
    monkeypatch.setitem(sys.modules, "openai", _make_fake_openai_module())
    _bypass_throttle(monkeypatch)

    from daccord.eval.clients import OpenAIClient
    from daccord.eval.schema import PromptMessages

    client = OpenAIClient(model="gpt-5-mini")
    resp = client.generate(PromptMessages(system="sys", user="usr"), run_id="r", batch_id="b")
    assert resp.parse_error is None
    assert resp.top1 is not None
    assert resp.top1.citation_id == "x"
    assert resp.input_tokens == 80
    assert resp.output_tokens == 40


def test_openai_client_json_decode_error_records_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_env_keys(monkeypatch)
    monkeypatch.setitem(sys.modules, "openai", _make_fake_openai_module(payload_json="not json {"))
    _bypass_throttle(monkeypatch)

    from daccord.eval.clients import OpenAIClient
    from daccord.eval.schema import PromptMessages

    client = OpenAIClient(model="gpt-5-mini")
    resp = client.generate(PromptMessages(system="sys", user="usr"), run_id="r", batch_id="b")
    assert resp.top1 is None
    assert resp.parse_error is not None
    assert "json decode" in resp.parse_error


def test_openai_client_api_error_surfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env_keys(monkeypatch)
    # Same trick as the anthropic test: keep the APIError class identity
    # consistent between the raised exception and the client's import.
    first = _make_fake_openai_module()
    server_down = first.APIError("server down")  # type: ignore[attr-defined]
    second = _make_fake_openai_module(raise_exc=server_down)
    second.APIError = first.APIError  # type: ignore[attr-defined]
    second.APITimeoutError = first.APITimeoutError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", second)
    _bypass_throttle(monkeypatch)

    from daccord.eval.clients import OpenAIClient
    from daccord.eval.schema import PromptMessages

    client = OpenAIClient(model="gpt-5-mini")
    resp = client.generate(PromptMessages(system="sys", user="usr"), run_id="r", batch_id="b")
    assert resp.top1 is None
    assert resp.parse_error is not None
    assert "server down" in resp.parse_error


def test_together_client_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """TogetherClient uses the openai SDK with a custom base_url."""
    _set_env_keys(monkeypatch)
    monkeypatch.setitem(sys.modules, "openai", _make_fake_openai_module())
    _bypass_throttle(monkeypatch)

    from daccord.eval.clients import TogetherClient
    from daccord.eval.schema import PromptMessages

    client = TogetherClient(
        model="Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
    )
    resp = client.generate(PromptMessages(system="sys", user="usr"), run_id="r", batch_id="b")
    assert resp.parse_error is None
    assert resp.top1 is not None
    assert resp.input_tokens == 80


def test_together_client_uses_together_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env_keys(monkeypatch)
    fake = _make_fake_openai_module()
    monkeypatch.setitem(sys.modules, "openai", fake)
    _bypass_throttle(monkeypatch)

    from daccord.eval.clients import TogetherClient

    client = TogetherClient(
        model="Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
    )
    # The fake OpenAI client preserves `base_url` from its constructor.
    assert client._client.base_url == "https://api.together.xyz/v1"  # type: ignore[attr-defined]


def test_anthropic_client_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Construction error is explicit, not deferred to the first call."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "anthropic", _make_fake_anthropic_module(tool_input={}))

    from daccord.eval.clients import AnthropicClient

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicClient(model="claude-haiku-4-5")


def test_openai_client_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "openai", _make_fake_openai_module())

    from daccord.eval.clients import OpenAIClient

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIClient(model="gpt-5-mini")


def test_together_client_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "openai", _make_fake_openai_module())

    from daccord.eval.clients import TogetherClient

    with pytest.raises(RuntimeError, match="TOGETHER_API_KEY"):
        TogetherClient(model="Qwen/Qwen3-235B-A22B-Instruct-2507-tput")
