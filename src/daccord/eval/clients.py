"""Provider adapters for the eval harness.

Adapters:

  - `GroqClient`     — Groq-hosted Llama / Qwen / Gemma (tier 2B)
  - `GeminiClient`   — Google Gemini via `google-genai` (tier 2B)
  - `RetrievalClient`— FAISS retrieval baseline (tier 12B; reused at serving)
  - `LocalHFClient`  — local 4-bit-NF4 Qwen3-8B baseline (tier 3A)

API clients (Groq, Gemini) use the provider's native JSON-schema
constraint:
  - Groq:   `response_format={"type": "json_object"}` (+ schema in the prompt)
  - Gemini: `config.response_schema=<pydantic model>` (first-class)

Local clients (Retrieval, LocalHF) bypass `daccord.costs.preflight` /
`record_call` — they have zero $-cost, and `daily.csv` is the spend log,
not a generic call ledger. Latency + token counts still flow on
`ModelResponse` (recorded to CSV + MLflow by the runner).

API spend tracking: every API call routes through `daccord.costs.preflight`
+ `record_call`. For free-tier providers, the daily cap is RPD; for
paid-spill (Anthropic/OpenAI/Together) the cap is USD/day. The contract
is identical from the caller.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from daccord.costs import preflight, record_call
from daccord.costs.config import Provider
from daccord.eval._rpm import api_throttle, gemini_retry_on_transient
from daccord.eval.retrieval_index import RetrievalIndexEntry, load_index
from daccord.eval.schema import CitationCandidate, ModelResponse, PromptMessages
from daccord.validation import validated

# Rough token estimate for preflight checks. Real token counts come back
# from the SDK response and are recorded post-call. The 4 chars/token
# heuristic is good enough for cap-headroom decisions on small prompts.
_CHARS_PER_TOKEN = 4
# Anthropic-style structured output schema we ask the model to fill.
# Kept inline here (not auto-derived from CitationCandidate) so the prompt
# documentation matches the wire format exactly — any field rename in the
# schema would force an intentional edit of this constant.
_CANDIDATE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "citation_id": {"type": "string"},
        "target_mechanism": {"type": "string"},
        "mapping_justification": {"type": "string"},
    },
    "required": ["citation_id", "target_mechanism", "mapping_justification"],
}


@runtime_checkable
class ModelClient(Protocol):
    """Single-method protocol for any generator backend used by the eval runner.

    `generate` returns a normalized `ModelResponse` regardless of provider.
    `provider` and `model` are surfaced as instance attributes so the runner
    can tag MLflow runs and CSV rows without re-deriving them.

    `@runtime_checkable` lets pydantic `@validated`-decorated functions
    accept `ModelClient` parameters via isinstance — needed by
    `daccord.serving.HybridRouter` which takes two ModelClients (retrieval +
    fine-tune) at construction.
    """

    provider: Provider
    model: str

    def generate(
        self, messages: PromptMessages, *, run_id: str, batch_id: str
    ) -> ModelResponse: ...


def _estimate_tokens(messages: PromptMessages) -> int:
    """Cheap char/4 heuristic. Used for preflight RPD checks (free-tier) and
    USD-cap estimation (paid fallback). Real token counts come from the SDK."""
    return (len(messages.system) + len(messages.user)) // _CHARS_PER_TOKEN


def _parse_candidate(raw_text: str) -> tuple[CitationCandidate | None, str | None]:
    """Parse a model's JSON output into a `CitationCandidate`.

    Returns `(candidate, None)` on success; `(None, error_msg)` on any
    JSON or schema failure. The runner records parse failures as Tier-1
    misses with the error surfaced in the CSV's judge_reasoning column.
    """
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return None, f"json decode at char {exc.pos}: {exc.msg}"
    if not isinstance(payload, dict):
        return None, f"expected JSON object, got {type(payload).__name__}"
    try:
        return CitationCandidate.model_validate(payload), None
    except Exception as exc:  # pydantic ValidationError or anything else
        return None, f"schema validation: {exc}"


class GroqClient:
    """Adapter for Groq-hosted OSS models (Llama, Qwen3, Gemma, etc.).

    Default model: `meta-llama/llama-4-scout-17b-16e-instruct` — current-
    generation free-tier Llama via Groq (17B active × 16E MoE). The same
    class also serves `qwen/qwen3-32b` when `--models qwen3` is selected;
    only the model string differs, the SDK call shape is identical.
    Note: when this class serves as both the `groq` generator and the
    `GroqJudge` judge in the same run, the result is technically self-judging
    on the `groq` row (a known M0 noise term).
    """

    provider: Provider = "groq"

    @validated
    def __init__(self, model: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> None:
        try:
            from groq import Groq  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover — dep is pinned
            raise RuntimeError("groq SDK not installed (uv sync)") from exc
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set — see .env.example")
        self.model = model
        self._client = Groq(api_key=api_key)

    @validated
    def generate(self, messages: PromptMessages, *, run_id: str, batch_id: str) -> ModelResponse:
        from groq import APIError  # type: ignore[import-not-found]

        est_in = _estimate_tokens(messages)
        # Generous budget: Qwen 3-32B and other "thinking" models can emit
        # ~1000-1500 tokens of <think>…</think> reasoning before the JSON
        # answer; max_tokens is a ceiling not a floor, so non-thinking
        # models still stop at their natural ~200-token completion.
        est_out = 2000
        preflight(self.provider, self.model, est_in, est_out)
        api_throttle()

        t0 = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": messages.system},
                    {"role": "user", "content": messages.user},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=est_out,
            )
        except APIError as exc:
            # Per-call failures (400 json_validate_failed, 429 rate-limit, etc.)
            # are recorded as Tier-1 misses rather than killing the whole run.
            # Preview models (Llama 4 Scout, Qwen 3-32B) occasionally return
            # empty completions that Groq's JSON validator rejects with 400.
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return ModelResponse(
                model=self.model,
                top1=None,
                raw_text="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                parse_error=f"groq api error: {type(exc).__name__}: {exc}",
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        raw_text = resp.choices[0].message.content or ""
        usage = resp.usage
        actual_in = int(usage.prompt_tokens) if usage else est_in
        actual_out = int(usage.completion_tokens) if usage else len(raw_text) // _CHARS_PER_TOKEN
        record_call(
            self.provider, self.model, actual_in, actual_out, run_id=run_id, batch_id=batch_id
        )

        candidate, parse_error = _parse_candidate(raw_text)
        return ModelResponse(
            model=self.model,
            top1=candidate,
            raw_text=raw_text,
            input_tokens=actual_in,
            output_tokens=actual_out,
            latency_ms=latency_ms,
            parse_error=parse_error,
        )


class RetrievalClient:
    """Retrieval baseline — top-1 cosine over a pre-built FAISS index.

    Local-only: no API call, bypasses `costs.preflight`/`record_call`.
    Zero citation hallucination by construction — the citation_id is
    copied verbatim from a hand-validated `GoldPair` indexed at build
    time.

    Consumes `PromptMessages.source_clause_text` + `target_jurisdiction`
    (populated by `build_eval_prompt`). The index is filtered to entries
    matching `target_jurisdiction` *before* the cosine search, so the
    retrieval scope is "nearest indexed clause whose mapping targets the
    requested jurisdiction" — matches the eval task shape.

    Threshold semantics: if `score_threshold` is set and the top-1
    cosine is below it, returns a `ModelResponse` with `top1=None` and
    `parse_error="no confident retrieval match (cosine=<...>, threshold=<...>)"`.
    The eval runner treats that as a Tier-1 miss + skipped Tier-2 judge
    (per existing `judge_pair` semantics for `top1=None`).

    The same index file is reused at serving time by
    `daccord.serving.hybrid.HybridRouter` — that's the whole point of
    co-locating it in the shared `daccord` package rather than the
    eval env.
    """

    provider: Provider = "retrieval"

    @validated
    def __init__(
        self,
        index_path: Path,
        embedder_name: str = "paraphrase-multilingual-mpnet-base-v2",
        score_threshold: float | None = None,
    ) -> None:
        try:
            from sentence_transformers import (  # type: ignore[import-not-found]
                SentenceTransformer,
            )
        except ImportError as exc:  # pragma: no cover — dep in envs/eval + consumer
            raise RuntimeError(
                "sentence-transformers not installed (envs/eval or consumer env)"
            ) from exc

        self.model = f"retrieval/{embedder_name}"
        self._embedder_name = embedder_name
        self._score_threshold = score_threshold
        self._embedder = SentenceTransformer(embedder_name)
        # `load_index` returns a faiss.Index typed as `object` because faiss
        # is a deferred import (no public type stubs). Narrow to Any here so
        # `.ntotal` + `.search(...)` don't trip pyright; the dynamic
        # attributes are exercised by test_retrieval_client.
        loaded_index, loaded_entries = load_index(index_path)
        self._index: Any = loaded_index
        self._entries: list[RetrievalIndexEntry] = loaded_entries
        # Pre-bucket entries by target_jurisdiction so filtering at query time
        # is O(1) lookup + O(k) cosine over the bucket, not O(n) over all
        # entries. For the gold-set sizes here (500–1000 entries) this is
        # cosmetic, but it lets the same client serve the much larger
        # consolidated indexes hybrid serving might use later without a
        # signature change.
        buckets: dict[str, list[int]] = {}
        for i, e in enumerate(self._entries):
            buckets.setdefault(e.target_jurisdiction, []).append(i)
        self._buckets = buckets

    @validated
    def generate(self, messages: PromptMessages, *, run_id: str, batch_id: str) -> ModelResponse:
        # Both fields are populated by build_eval_prompt; defensive guard
        # because PromptMessages allows them as None for API clients.
        if messages.source_clause_text is None or messages.target_jurisdiction is None:
            return ModelResponse(
                model=self.model,
                top1=None,
                raw_text="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=0.0,
                parse_error=(
                    "retrieval requires PromptMessages.source_clause_text and "
                    "target_jurisdiction — caller did not populate them"
                ),
            )

        bucket = self._buckets.get(messages.target_jurisdiction, [])
        if not bucket:
            return ModelResponse(
                model=self.model,
                top1=None,
                raw_text="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=0.0,
                parse_error=(
                    f"no indexed entries for target_jurisdiction={messages.target_jurisdiction!r}"
                ),
            )

        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("numpy not installed (envs/eval or consumer env)") from exc

        t0 = time.perf_counter()
        raw_q = self._embedder.encode(
            [messages.source_clause_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        query = np.ascontiguousarray(raw_q, dtype=np.float32)
        # IndexFlatIP returns scores in descending cosine order (because
        # vectors were L2-normalized at build time). We pull top-K=len(bucket)
        # across the *whole* index, then walk the result in order keeping
        # the first hit that belongs to our jurisdiction bucket — cheap
        # for the sizes here and avoids building a per-jurisdiction FAISS
        # sub-index.
        top_k = min(self._index.ntotal, max(8, len(bucket)))
        scores, ids = self._index.search(query, top_k)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        bucket_set = set(bucket)
        best_score: float | None = None
        best_entry: RetrievalIndexEntry | None = None
        for sc, idx in zip(scores[0], ids[0], strict=True):
            if int(idx) in bucket_set:
                best_score = float(sc)
                best_entry = self._entries[int(idx)]
                break

        if best_entry is None or best_score is None:
            return ModelResponse(
                model=self.model,
                top1=None,
                raw_text="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                parse_error=(
                    f"no hit in target_jurisdiction={messages.target_jurisdiction!r} "
                    f"within top {top_k}"
                ),
            )

        if self._score_threshold is not None and best_score < self._score_threshold:
            return ModelResponse(
                model=self.model,
                top1=None,
                raw_text=json.dumps(
                    {
                        "gold_id": best_entry.gold_id,
                        "cosine": best_score,
                        "threshold": self._score_threshold,
                    }
                ),
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                parse_error=(
                    f"no confident retrieval match (cosine={best_score:.4f}, "
                    f"threshold={self._score_threshold:.4f})"
                ),
            )

        candidate = CitationCandidate(
            citation_id=best_entry.target_citation_id,
            target_mechanism=best_entry.target_mechanism,
            # Honest justification: this answer comes from a validated
            # gold pair, not from generative reasoning. Includes cosine
            # so a consumer can show confidence.
            mapping_justification=(
                f"Retrieved verbatim from gold pair {best_entry.gold_id} (cosine={best_score:.4f})"
            ),
        )
        raw_text = json.dumps(
            {
                "gold_id": best_entry.gold_id,
                "cosine": best_score,
                "citation_id": candidate.citation_id,
                "target_mechanism": candidate.target_mechanism,
                "mapping_justification": candidate.mapping_justification,
            }
        )
        return ModelResponse(
            model=self.model,
            top1=candidate,
            raw_text=raw_text,
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
        )


class GeminiClient:
    """Adapter for Google Gemini (via `google-genai`).

    Default model: `gemini-3.1-flash-lite` — free-tier 15 RPM / 500 RPD;
    native JSON-schema constrained output. (Older `gemini-2.5-flash` was
    dropped — its free-tier daily cap was 20 RPD on some accounts.)
    """

    provider: Provider = "google_gemini"

    @validated
    def __init__(self, model: str = "gemini-3.1-flash-lite") -> None:
        try:
            from google import genai  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("google-genai SDK not installed (uv sync)") from exc
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set — see .env.example")
        self.model = model
        self._client = genai.Client(api_key=api_key)

    @validated
    def generate(self, messages: PromptMessages, *, run_id: str, batch_id: str) -> ModelResponse:
        from google.genai import types  # type: ignore[import-not-found]

        est_in = _estimate_tokens(messages)
        est_out = 400
        preflight(self.provider, self.model, est_in, est_out)
        api_throttle()

        config = types.GenerateContentConfig(
            system_instruction=messages.system,
            temperature=0.0,
            max_output_tokens=est_out,
            response_mime_type="application/json",
            response_json_schema=_CANDIDATE_JSON_SCHEMA,
        )
        t0 = time.perf_counter()
        resp = gemini_retry_on_transient(
            lambda: self._client.models.generate_content(
                model=self.model,
                contents=messages.user,
                config=config,
            )
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        raw_text = resp.text or ""
        usage = resp.usage_metadata
        actual_in = int(getattr(usage, "prompt_token_count", None) or est_in)
        actual_out = int(
            getattr(usage, "candidates_token_count", None) or (len(raw_text) // _CHARS_PER_TOKEN)
        )
        record_call(
            self.provider, self.model, actual_in, actual_out, run_id=run_id, batch_id=batch_id
        )

        candidate, parse_error = _parse_candidate(raw_text)
        return ModelResponse(
            model=self.model,
            top1=candidate,
            raw_text=raw_text,
            input_tokens=actual_in,
            output_tokens=actual_out,
            latency_ms=latency_ms,
            parse_error=parse_error,
        )


class LocalHFClient:
    """Local 4-bit-NF4 Qwen3-8B baseline (tier 3A).

    Loaded via HuggingFace `transformers` + `bitsandbytes`. Quantisation
    deliberately matches the load condition the tier 10–12 QLoRA training
    uses (NF4 + bfloat16 compute + double-quant) so the M4 fine-tune
    delta is apples-to-apples against the actual production-shape base.

    Default base: `Qwen/Qwen3-8B` (Apr 2025). The earlier `Qwen2.5-7B-Instruct`
    was the project's locked base at tier 2C tokenizer-audit time; revisited
    on 2026-05-25 in favour of Qwen 3 (newer multilingual tokenizer, similar
    VRAM footprint at NF4).

    Local-only: bypasses `costs.preflight` / `record_call` (zero $-cost;
    `daily.csv` is the spend log, not a generic call ledger). Latency and
    token counts still flow on `ModelResponse` and land in CSV + MLflow
    via the runner.

    No native JSON-schema constraint. Output discipline is prompt-only;
    parse failures degrade to Tier-1 misses with the error surfaced in
    `judge_reasoning` by the runner.
    """

    provider: Provider = "local_hf"

    @validated
    def __init__(
        self,
        model: str = "Qwen/Qwen3-8B",
        quantization: Literal["nf4", "bf16"] = "nf4",
        max_new_tokens: int = 400,
    ) -> None:
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:  # pragma: no cover — dep in envs/baseline
            raise RuntimeError(
                "transformers / torch not installed (use envs/baseline service)"
            ) from exc

        self.model = model
        self._max_new_tokens = max_new_tokens
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        if quantization == "nf4":
            try:
                import bitsandbytes  # noqa: F401  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover — dep in envs/baseline
                raise RuntimeError(
                    "bitsandbytes not installed (use envs/baseline service)"
                ) from exc
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model,
                quantization_config=bnb_cfg,
                device_map="auto",
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

    @validated
    def generate(self, messages: PromptMessages, *, run_id: str, batch_id: str) -> ModelResponse:
        import torch  # type: ignore[import-not-found]

        rendered = self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": messages.system},
                {"role": "user", "content": messages.user},
            ],
            tokenize=False,
            add_generation_prompt=True,
            # Qwen 3 introduced thinking mode (prepends <think>…</think> before
            # the JSON answer) — disable it so the response starts at the JSON
            # opening brace. Unknown kwargs are ignored by templates that
            # don't reference them (no-op for older bases).
            enable_thinking=False,
        )
        encoded = self._tokenizer(rendered, return_tensors="pt").to(self._model.device)
        prompt_len = int(encoded["input_ids"].shape[1])

        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = self._model.generate(
                **encoded,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        new_token_ids = out_ids[0, prompt_len:]
        completion_len = int(new_token_ids.shape[0])
        raw_text = self._tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

        candidate, parse_error = _parse_candidate(raw_text)
        return ModelResponse(
            model=self.model,
            top1=candidate,
            raw_text=raw_text,
            input_tokens=prompt_len,
            output_tokens=completion_len,
            latency_ms=latency_ms,
            parse_error=parse_error,
        )
