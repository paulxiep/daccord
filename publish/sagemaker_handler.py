"""SageMaker custom inference handler — wraps `HybridRouter` for endpoint serving.

Loaded by SageMaker's PyTorch / HuggingFace inference container at endpoint
start time. SageMaker invokes the four standard hooks:

    model_fn(model_dir)      → load adapter + retrieval index + embedder; return state
    input_fn(request, type)  → parse JSON payload into a GoldPair query
    predict_fn(state, query) → route via HybridRouter
    output_fn(prediction, t) → serialize HybridResponse to JSON

The payload contract (request/response JSON) is the API surface Pillar C's
`cross_jurisdiction_map` tool consumes — see [aws_credential_plan.md](../aws_credential_plan.md)
§Pillar C Architecture. Keeping it minimal + stable:

Request:
```
{
  "source_jurisdiction": str,
  "source_framework":   str,
  "source_citation_id": str,
  "source_clause":      str,
  "target_jurisdiction": str,
  "target_framework":   str
}
```

Response:
```
{
  "candidate": { "citation_id": str, "target_provision": str, "justification": str } | null,
  "provenance": "gold-retrieval" | "fine-tune-generalization" | "no-confident-match",
  "retrieval": { "model": str, "latency_ms": float, "parse_error": str | null },
  "fine_tune": { "model": str, "latency_ms": float, "parse_error": str | null } | null
}
```

The agent in Pillar C branches on `provenance` to decide whether to call
`verify_citation` (skip on `gold-retrieval`, trigger on
`fine-tune-generalization`, fall through to `search_regulation_text` on
`no-confident-match`).

`model_dir` layout (produced by `publish/package_model.py`):
```
<model_dir>/
  code/
    inference.py           ← this file, renamed (SageMaker convention)
    requirements.txt       ← runtime deps
  adapter/                 ← QLoRA adapter from tier 12A
  retrieval/
    index.faiss            ← FAISS index over train-split source clauses
    gold_pairs.jsonl       ← parallel RetrievalIndexEntry rows
  embedder/                ← MPNet snapshot (no internet at endpoint)
```
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daccord.eval.clients import RetrievalClient
from daccord.eval.schema import CitationCandidate, ModelResponse
from daccord.gold import GoldPair
from daccord.serving import HybridResponse, HybridRouter, LocalAdapterClient
from daccord.validation import ValidatedModel, validated

# SageMaker requires this content-type be supported as input + output.
JSON_CONTENT_TYPE = "application/json"

# Retrieval threshold tuned at deploy time. Default = 90th percentile of
# train-split self-similarity (see plan Part 5.3). Override via the
# SAGEMAKER_RETRIEVAL_THRESHOLD env var on the endpoint config.
DEFAULT_RETRIEVAL_THRESHOLD = 0.75


class HandlerState(ValidatedModel):
    """In-memory state loaded once at endpoint cold start."""

    router: HybridRouter
    model_dir: str


class _RequestPayload(ValidatedModel):
    """Shape of the inbound SageMaker request JSON."""

    source_jurisdiction: str
    source_framework: str
    source_citation_id: str
    source_clause: str
    target_jurisdiction: str
    target_framework: str


@validated
def _resolve_threshold() -> float:
    import os

    raw = os.environ.get("SAGEMAKER_RETRIEVAL_THRESHOLD")
    if raw is None:
        return DEFAULT_RETRIEVAL_THRESHOLD
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"SAGEMAKER_RETRIEVAL_THRESHOLD={raw!r} is not a float") from exc


@validated
def model_fn(model_dir: str) -> HandlerState:
    """SageMaker entry-point — load adapter + retrieval index at cold start.

    Called once per endpoint container at startup. Heavy load (~10–30s):
    reads the QLoRA adapter into VRAM, loads the MPNet embedder, mmap's
    the FAISS index. Subsequent `predict_fn` calls are fast.

    Embedder is loaded from a local snapshot (no internet at endpoint), so
    `package_model.py` must include the embedder model files alongside the
    index.
    """
    root = Path(model_dir)
    retrieval_dir = root / "retrieval"
    adapter_dir = root / "adapter"
    embedder_dir = root / "embedder"

    # The RetrievalClient's index_path takes a stem; .faiss + .jsonl are derived.
    retrieval_client = RetrievalClient(
        index_path=retrieval_dir / "index",
        embedder_name=str(embedder_dir),
        score_threshold=_resolve_threshold(),
    )
    fine_tune_client = LocalAdapterClient(adapter_path=adapter_dir)
    router = HybridRouter(retrieval_client=retrieval_client, fine_tune_client=fine_tune_client)
    return HandlerState(router=router, model_dir=str(root))


@validated
def input_fn(request_body: str, content_type: str) -> GoldPair:
    """Parse SageMaker request JSON into a query `GoldPair`.

    Builds a synthetic `GoldPair` with empty target_* fields — only the
    source_* fields + target_jurisdiction matter at query time
    (`build_eval_prompt` doesn't include the target citation/mechanism
    in the rendered prompt, by design — see eval/prompts.py).
    """
    if content_type != JSON_CONTENT_TYPE:
        raise ValueError(
            f"unsupported content_type {content_type!r}; only {JSON_CONTENT_TYPE} supported"
        )
    payload = _RequestPayload.model_validate_json(request_body)
    return GoldPair(
        id="endpoint-query",
        source_jurisdiction=payload.source_jurisdiction,
        source_framework=payload.source_framework,
        source_citation_id=payload.source_citation_id,
        source_mechanism=payload.source_clause,
        source_language="",  # not consumed by prompt builder
        target_jurisdiction=payload.target_jurisdiction,
        target_framework=payload.target_framework,
        target_citation_id="",  # never given to the model (would leak the answer)
        target_mechanism="",
        target_language="",
        notes=None,
    )


@validated
def predict_fn(query: GoldPair, state: HandlerState) -> HybridResponse:
    """Route the query through HybridRouter. Stateless besides `state.router`."""
    return state.router.route(query, run_id="endpoint", batch_id=query.id)


@validated
def output_fn(response: HybridResponse, accept: str) -> tuple[str, str]:
    """Serialize HybridResponse to the public JSON contract.

    Returns `(body, content_type)` — SageMaker's expected signature.
    """
    if accept not in (JSON_CONTENT_TYPE, "*/*"):
        raise ValueError(f"unsupported accept {accept!r}; only {JSON_CONTENT_TYPE} supported")
    payload: dict[str, Any] = {
        "candidate": (
            None
            if response.candidate is None
            else {
                "citation_id": response.candidate.citation_id,
                "target_provision": response.candidate.target_mechanism,
                "justification": response.candidate.mapping_justification,
            }
        ),
        "provenance": response.provenance,
        "retrieval": _summarize(response.retrieval_response),
        "fine_tune": (
            None if response.fine_tune_response is None else _summarize(response.fine_tune_response)
        ),
    }
    return json.dumps(payload), JSON_CONTENT_TYPE


def _summarize(resp: ModelResponse) -> dict[str, Any]:
    """Surface only what Pillar C's agent needs — no token counts / raw_text."""
    return {
        "model": resp.model,
        "latency_ms": resp.latency_ms,
        "parse_error": resp.parse_error,
    }


# Local invocation helpers — useful for unit-testing the handler without
# spinning up a SageMaker endpoint. Not called by SageMaker itself.


@validated
def invoke_local(state: HandlerState, request_body: str) -> str:
    """Run input → predict → output end-to-end. Returns the JSON body."""
    query = input_fn(request_body, JSON_CONTENT_TYPE)
    prediction = predict_fn(query, state)
    body, _ = output_fn(prediction, JSON_CONTENT_TYPE)
    return body


__all__ = [
    "JSON_CONTENT_TYPE",
    "CitationCandidate",
    "HandlerState",
    "input_fn",
    "invoke_local",
    "model_fn",
    "output_fn",
    "predict_fn",
]
