"""Hybrid router — retrieval-first, QLoRA-fallback, provenance-tagged.

The router composes two `ModelClient`s (from `daccord.eval.clients`) so the
eval-time and serving-time client implementations are the same. Routing logic:

  1. Build a `PromptMessages` from `(source_clause, target_jurisdiction)`
     (delegates to `build_eval_prompt` for prompt template parity with
     M0/M4 eval).
  2. Call the retrieval client. If it returns a non-None `top1`, that's
     `gold-retrieval` provenance — return it.
  3. Otherwise call the fine-tune client. If it returns a non-None `top1`,
     that's `fine-tune-generalization` provenance.
  4. Otherwise both failed — `no-confident-match`.

`HybridResponse` exposes both underlying responses so callers (Streamlit
consumer, SageMaker handler) can inspect retrieval cosine + fine-tune
latency + parse_errors independently of which path returned the answer.
"""

from __future__ import annotations

from typing import Literal

from daccord.eval.clients import ModelClient
from daccord.eval.schema import CitationCandidate, ModelResponse
from daccord.gold import GoldPair
from daccord.validation import ValidatedModel, validated

Provenance = Literal[
    "gold-retrieval",
    "fine-tune-generalization",
    "no-confident-match",
]


class HybridResponse(ValidatedModel):
    """Result of one hybrid-routed call.

    `candidate` is the answer returned to the caller (None when no path
    produced one). `provenance` says which path produced it. The two
    underlying `ModelResponse`s are preserved so the consumer can show
    cosine + latency + parse_error per path even when only one path's
    answer was surfaced.

    `fine_tune_response` is `None` when retrieval succeeded (we don't
    invoke fine-tune unnecessarily — saves SageMaker endpoint compute).
    """

    candidate: CitationCandidate | None
    provenance: Provenance
    retrieval_response: ModelResponse
    fine_tune_response: ModelResponse | None = None


class HybridRouter:
    """Retrieval-first, QLoRA-fallback inference with provenance tagging.

    Construction wires two `ModelClient`s; routing logic is in `route()`.
    Both clients receive the *same* `PromptMessages` so they can be
    composed into the eval harness for an A/B comparison (e.g. side-by-side
    consumer view).

    Caller responsibilities:
      - The `retrieval_client` should be a `RetrievalClient` configured
        with a meaningful `score_threshold` — without one, retrieval
        always "succeeds" by returning the closest gold pair regardless of
        cosine, which defeats the hybrid routing's purpose.
      - The `fine_tune_client` should be a `LocalAdapterClient` (or any
        `ModelClient` that wraps the QLoRA adapter). At demo time before
        an adapter exists, pass any ModelClient — failures surface as
        `no-confident-match`.

    Why the router lives in `daccord.serving` and not `daccord.eval`:
    the eval harness consumes `ModelClient`s individually for fair
    per-baseline scoring; the router *combines* them and is only used at
    serving time. Mixing the two would tangle the comparator semantics.
    """

    @validated
    def __init__(
        self,
        retrieval_client: ModelClient,
        fine_tune_client: ModelClient,
    ) -> None:
        self._retrieval = retrieval_client
        self._fine_tune = fine_tune_client

    @validated
    def route(
        self,
        gold_pair: GoldPair,
        *,
        run_id: str,
        batch_id: str,
    ) -> HybridResponse:
        """Route one (source_clause, target_jurisdiction) lookup.

        Takes a `GoldPair` rather than raw `(str, str)` so the prompt
        builder + retrieval's jurisdiction-filter signals come from a
        single typed source. Callers constructing ad-hoc lookups (e.g.
        SageMaker request payloads) build a synthetic `GoldPair` with
        empty `target_*` fields — those fields are ignored on the query
        side; only `source_*` and `target_jurisdiction` matter.
        """
        # Import locally to avoid eval-namespace circular import; prompts
        # depends on schema which has no upward dep on serving.
        from daccord.eval.prompts import build_eval_prompt

        prompt = build_eval_prompt(gold_pair)
        retrieval_resp = self._retrieval.generate(prompt, run_id=run_id, batch_id=batch_id)
        if retrieval_resp.top1 is not None:
            return HybridResponse(
                candidate=retrieval_resp.top1,
                provenance="gold-retrieval",
                retrieval_response=retrieval_resp,
                fine_tune_response=None,
            )

        fine_tune_resp = self._fine_tune.generate(prompt, run_id=run_id, batch_id=batch_id)
        if fine_tune_resp.top1 is not None:
            return HybridResponse(
                candidate=fine_tune_resp.top1,
                provenance="fine-tune-generalization",
                retrieval_response=retrieval_resp,
                fine_tune_response=fine_tune_resp,
            )

        return HybridResponse(
            candidate=None,
            provenance="no-confident-match",
            retrieval_response=retrieval_resp,
            fine_tune_response=fine_tune_resp,
        )
