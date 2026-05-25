"""Mocked tests for LocalHFClient — CI-runnable without a GPU.

Stubs `transformers.AutoModelForCausalLM` / `AutoTokenizer` /
`BitsAndBytesConfig` + `bitsandbytes` + the bits of `torch` the client
touches at import time. Exercises construction (NF4 + bf16 paths),
generation happy path, and parse-failure degradation.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest


class _FakeTensor:
    """Minimal tensor stand-in covering `.shape`, `.to(...)`, and the
    `out_ids[0, prompt_len:]` slice the client uses to count new tokens."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def to(self, _device: Any) -> _FakeTensor:
        return self

    def __getitem__(self, item: Any) -> _FakeTensor:
        # Only branch the client uses: `out_ids[0, prompt_len:]` — drops the
        # batch dim and returns a 1-D tensor of (seq_len - prompt_len,).
        if isinstance(item, tuple) and len(item) == 2:
            _row, sl = item
            start = sl.start if isinstance(sl, slice) else 0
            new_len = self.shape[1] - start
            return _FakeTensor((new_len,))
        return self


class _FakeEncoded(dict):  # type: ignore[type-arg]
    """Tokenizer call result. dict so `**encoded` unpacks into model.generate."""

    def __init__(self, prompt_len: int) -> None:
        super().__init__()
        self["input_ids"] = _FakeTensor((1, prompt_len))
        self["attention_mask"] = _FakeTensor((1, prompt_len))

    def to(self, _device: Any) -> _FakeEncoded:
        return self


class _FakeTokenizer:
    eos_token_id = 0

    def __init__(self) -> None:
        self._decoded = ""

    def apply_chat_template(
        self,
        _messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **_extra: object,
    ) -> str:
        # The LocalHFClient passes Qwen 3's `enable_thinking=False` kwarg;
        # real tokenizer templates ignore unknown kwargs, so the fake must too.
        assert tokenize is False and add_generation_prompt is True
        return "rendered prompt"

    def __call__(self, _text: str, *, return_tensors: str) -> _FakeEncoded:
        assert return_tensors == "pt"
        return _FakeEncoded(prompt_len=5)

    def decode(self, _ids: Any, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return self._decoded


class _FakeModel:
    def __init__(self) -> None:
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, _model: str, **_kwargs: Any) -> _FakeModel:
        return cls()

    def generate(self, **_kwargs: Any) -> _FakeTensor:
        # 5-token prompt + 8-token completion → out_ids.shape = (1, 13)
        return _FakeTensor((1, 13))


@pytest.fixture
def fake_transformers(monkeypatch: pytest.MonkeyPatch) -> _FakeTokenizer:
    """Inject fake transformers + bitsandbytes + minimal torch into sys.modules."""
    transformers_mod = types.ModuleType("transformers")
    tokenizer_instance = _FakeTokenizer()
    tokenizer_instance._decoded = (
        '{"citation_id": "Art. 5", "target_mechanism": "lawful basis", '
        '"mapping_justification": "matches gdpr art 5 directly"}'
    )

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, _model: str) -> _FakeTokenizer:
            return tokenizer_instance

    transformers_mod.AutoTokenizer = _AutoTokenizer  # type: ignore[attr-defined]
    transformers_mod.AutoModelForCausalLM = _FakeModel  # type: ignore[attr-defined]

    class _BnbCfg:
        def __init__(self, **kw: Any) -> None:
            self.kw = kw

    transformers_mod.BitsAndBytesConfig = _BnbCfg  # type: ignore[attr-defined]

    bnb_mod = types.ModuleType("bitsandbytes")

    torch_mod = types.ModuleType("torch")
    torch_mod.bfloat16 = "bfloat16-sentinel"  # type: ignore[attr-defined]

    class _NoGrad:
        def __enter__(self) -> None: ...
        def __exit__(self, *_a: Any) -> None: ...

    torch_mod.no_grad = lambda: _NoGrad()  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    monkeypatch.setitem(sys.modules, "bitsandbytes", bnb_mod)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    return tokenizer_instance


def test_init_nf4_loads_quantized(fake_transformers: _FakeTokenizer) -> None:
    from daccord.eval.clients import LocalHFClient

    client = LocalHFClient(model="Qwen/Qwen3-8B", quantization="nf4")
    assert client.provider == "local_hf"
    assert client.model == "Qwen/Qwen3-8B"


def test_init_bf16_skips_bnb(fake_transformers: _FakeTokenizer) -> None:
    from daccord.eval.clients import LocalHFClient

    client = LocalHFClient(model="Qwen/Qwen3-8B", quantization="bf16")
    assert client.model == "Qwen/Qwen3-8B"


def test_generate_happy_path(fake_transformers: _FakeTokenizer) -> None:
    from daccord.eval.clients import LocalHFClient
    from daccord.eval.schema import PromptMessages

    client = LocalHFClient(model="Qwen/Qwen3-8B", quantization="nf4")
    resp = client.generate(PromptMessages(system="sys", user="usr"), run_id="rid", batch_id="bid")
    assert resp.parse_error is None
    assert resp.top1 is not None
    assert resp.top1.citation_id == "Art. 5"
    assert resp.input_tokens == 5
    assert resp.output_tokens == 8
    assert resp.model == "Qwen/Qwen3-8B"


def test_generate_parse_failure_records_error(fake_transformers: _FakeTokenizer) -> None:
    from daccord.eval.clients import LocalHFClient
    from daccord.eval.schema import PromptMessages

    fake_transformers._decoded = "not json at all"
    client = LocalHFClient(model="Qwen/Qwen3-8B", quantization="nf4")
    resp = client.generate(PromptMessages(system="sys", user="usr"), run_id="rid", batch_id="bid")
    assert resp.top1 is None
    assert resp.parse_error is not None
    assert "json" in resp.parse_error.lower()
