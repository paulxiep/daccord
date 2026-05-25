"""Provider adapters for the eval harness.

Two adapters ship at tier 2B for the free-tier OSS direction the project
took after the costs-config v2 (paid + free_tier) refactor:

  - `GroqClient`    — Groq-hosted Llama / Qwen / Gemma models
  - `GeminiClient`  — Google Gemini via `google-genai`

Both use the provider's native JSON-schema constraint:
  - Groq:   `response_format={"type": "json_object"}` (+ schema in the prompt)
  - Gemini: `config.response_schema=<pydantic model>` (first-class)

A `LocalHFClient` for the local Qwen2.5-7B-Instruct baseline is intentionally
NOT shipped at 2B — `torch` + `bitsandbytes` are not yet in the project
deps, and Groq does not host Qwen2.5-7B (only Qwen3-32B / Qwen-Coder), so
the base-Qwen baseline must run locally when tier 3A captures it. That
tier owns the LocalHF dep + adapter; the `ModelClient` Protocol below is
the contract it implements.

Every API call routes through [daccord.costs.preflight] + [record_call].
For free-tier providers, the daily cap is RPD (requests-per-day) and
`estimate_cost` returns 0.0; for paid-spill fallback (Anthropic/OpenAI/
Together) the cap is USD/day. The contract is identical from the caller.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from daccord.costs import preflight, record_call
from daccord.costs.config import Provider
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

    Default model: `llama-3.3-70b-versatile` — strongest free-tier general
    LLM at the time of writing; 1k RPD on the published free tier (the
    project's costs config sets a higher org-wide RPD cap; tighter
    per-model limits live with the provider).
    """

    provider: Provider = "groq"

    @validated
    def __init__(self, model: str = "llama-3.3-70b-versatile") -> None:
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
        est_in = _estimate_tokens(messages)
        est_out = 400  # observed ceiling for a single CitationCandidate JSON
        preflight(self.provider, self.model, est_in, est_out)

        t0 = time.perf_counter()
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

    Default model: `gemini-2.5-flash` — fast, free-tier (1500 RPD per the
    project's costs config), native JSON-schema constrained output.
    """

    provider: Provider = "google_gemini"

    @validated
    def __init__(self, model: str = "gemini-2.5-flash") -> None:
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

        config = types.GenerateContentConfig(
            system_instruction=messages.system,
            temperature=0.0,
            max_output_tokens=est_out,
            response_mime_type="application/json",
            response_json_schema=_CANDIDATE_JSON_SCHEMA,
        )
        t0 = time.perf_counter()
        resp = self._client.models.generate_content(
            model=self.model,
            contents=messages.user,
            config=config,
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
