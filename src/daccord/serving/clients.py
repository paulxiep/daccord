"""Fine-tune serving clients.

Companion to `daccord.eval.clients.RetrievalClient` — same `ModelClient`
Protocol contract, but generates from a QLoRA adapter rather than a
retrieval index.

`LocalAdapterClient` is the production implementation: loads a saved
QLoRA adapter via PEFT + transformers + bitsandbytes (4-bit quantization
for ~7B inference on consumer GPUs / the SageMaker `ml.g5.xlarge`
endpoint). Deferred-imports the heavy ML deps so the shared `daccord`
package stays importable in environments that don't carry them.

At demo time before the tier-12A QLoRA train produces an adapter, no
adapter exists; `LocalAdapterClient` will raise at construction. The
`HybridRouter` is designed to handle that gracefully — tests use mocked
ModelClients implementing the same shape.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from daccord.costs.config import Provider
from daccord.eval.clients import ModelClient as FineTuneClient
from daccord.eval.schema import CitationCandidate, ModelResponse, PromptMessages
from daccord.validation import validated

# `FineTuneClient` is a rename of `daccord.eval.clients.ModelClient` — identical
# Protocol shape. Re-exported here so future fine-tune client variants
# (e.g. `SageMakerRuntimeClient` for in-cluster inference, or an mlflow
# pyfunc wrapper) live in the `daccord.serving` namespace alongside
# `LocalAdapterClient` rather than being scattered between eval + serving.


class LocalAdapterClient:
    """In-process QLoRA adapter inference via PEFT + transformers.

    Loads at construction (slow — model weights + tokenizer + adapter
    must fit in VRAM); subsequent `generate()` calls are fast.

    Designed to be the same `ModelClient` Protocol shape as
    `GroqClient` / `GeminiClient` / `RetrievalClient` so the eval harness
    can score it as a fourth baseline at tier 12B/13 and the
    `HybridRouter` can compose it without an adapter shim.

    Local-only: bypasses the costs layer (provider="retrieval" repurposed
    here as the "no API spend" marker — adapter inference is free).
    A dedicated `"local_hf"` provider value can be added to the
    `costs.config.Provider` Literal later if a finer distinction is
    needed for telemetry.

    At demo time before tier 12A produces an adapter, `adapter_path`
    will not exist; construction raises with a friendly error pointing
    at the training tier.
    """

    provider: Provider = "retrieval"  # local-only; reuses the bypass-cost-layer marker

    @validated
    def __init__(
        self,
        adapter_path: Path,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens: int = 400,
        load_in_4bit: bool = True,
    ) -> None:
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"QLoRA adapter not found at {adapter_path}. "
                "Train one via tier 12A (`training/train.py`) and pass its "
                "output directory here. Until then, the consumer demo and "
                "SageMaker handler can run retrieval-only via the "
                "RetrievalClient; HybridRouter will tag fine-tune calls as "
                "`no-confident-match` if no adapter is wired."
            )

        try:
            import torch  # type: ignore[import-not-found]
            from peft import PeftModel  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:  # pragma: no cover — consumer/training env carries these
            raise RuntimeError(
                "LocalAdapterClient requires torch + transformers + peft + bitsandbytes "
                "(install in consumer/ or envs/training/)"
            ) from exc

        self.model = f"qlora-adapter:{adapter_path.name}"
        self._adapter_path = adapter_path
        self._base_model_name = base_model
        self._max_new_tokens = max_new_tokens

        quant_config: Any = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self._tokenizer = AutoTokenizer.from_pretrained(base_model)
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self._model = PeftModel.from_pretrained(base, str(adapter_path))
        self._model.eval()

    @validated
    def generate(self, messages: PromptMessages, *, run_id: str, batch_id: str) -> ModelResponse:
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch not installed") from exc

        # Use the model's chat template so the QLoRA-fine-tuned model
        # sees the same prompt format it was trained on. Qwen2.5-Instruct
        # has a standard chat template that transformers handles.
        chat = [
            {"role": "system", "content": messages.system},
            {"role": "user", "content": messages.user},
        ]
        prompt_text = self._tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._model.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Strip the prompt prefix; decode only the newly generated tokens.
        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        raw_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        input_token_count = int(inputs["input_ids"].shape[1])
        output_token_count = int(new_tokens.shape[0])

        # Reuse the eval-layer parser so JSON-shape validation is consistent.
        from daccord.eval.clients import _parse_candidate

        candidate, parse_error = _parse_candidate(raw_text)
        return ModelResponse(
            model=self.model,
            top1=candidate,
            raw_text=raw_text,
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            latency_ms=latency_ms,
            parse_error=parse_error,
        )


# CitationCandidate is re-exported so the SageMaker handler can construct
# synthetic responses (e.g. health-check) without reaching into the eval
# namespace.
__all__ = ["CitationCandidate", "FineTuneClient", "LocalAdapterClient"]
