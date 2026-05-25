"""HybridRouter routing-logic tests.

Network-free, ML-free: both `ModelClient`s are fakes that return
hard-coded `ModelResponse`s. Validates the three-branch routing:

  1. Retrieval hits → provenance="gold-retrieval"; fine-tune NOT called.
  2. Retrieval misses, fine-tune hits → provenance="fine-tune-generalization".
  3. Both miss → provenance="no-confident-match"; both responses surfaced.

Also pins one invariant: `HybridRouter.route()` builds the prompt via
`build_eval_prompt`, so both retrieval and fine-tune see PromptMessages
with `source_clause_text` + `target_jurisdiction` populated (the
RetrievalClient depends on these — see eval test_retrieval_client.py).
"""

from __future__ import annotations

from daccord.costs.config import Provider
from daccord.eval.schema import CitationCandidate, ModelResponse, PromptMessages
from daccord.gold import GoldPair
from daccord.serving import HybridRouter

GOLD = GoldPair(
    id="hr-001",
    source_jurisdiction="eu",
    source_framework="gdpr",
    source_citation_id="Art. 32",
    source_mechanism="Security of processing.",
    source_language="en",
    target_jurisdiction="sg",
    target_framework="pdpa_sg",
    target_citation_id="Section 24",
    target_mechanism="Reasonable security arrangements.",
    target_language="en",
    notes=None,
)

RETRIEVAL_HIT = CitationCandidate(
    citation_id="Section 24",
    target_mechanism="Reasonable security arrangements.",
    mapping_justification="Retrieved from gold pair hr-001 (cosine=0.95)",
)
FINE_TUNE_HIT = CitationCandidate(
    citation_id="Section 24",
    target_mechanism="Protection of personal data — security arrangements.",
    mapping_justification="Both impose a general security obligation on data controllers.",
)


def _resp(
    top1: CitationCandidate | None,
    model: str,
    parse_error: str | None = None,
) -> ModelResponse:
    return ModelResponse(
        model=model,
        top1=top1,
        raw_text="" if top1 is None else "{...}",
        input_tokens=10,
        output_tokens=20,
        latency_ms=1.0,
        parse_error=parse_error,
    )


class _RecordingClient:
    """Fake ModelClient that records calls + returns a fixed response.

    Class-level `provider: Provider` + `model: str` annotations are required
    for pyright to accept this as a structural match for the `ModelClient`
    Protocol — protocol attributes are invariant, so an inferred narrower
    type (e.g. `Literal["retrieval"]`) from a default value wouldn't match.
    """

    provider: Provider
    model: str

    def __init__(self, provider: Provider, model: str, response: ModelResponse) -> None:
        self.provider = provider
        self.model = model
        self._response = response
        self.calls: list[PromptMessages] = []

    def generate(self, messages: PromptMessages, *, run_id: str, batch_id: str) -> ModelResponse:
        self.calls.append(messages)
        return self._response


class TestHybridRouterRouting:
    def test_retrieval_hit_returns_gold_retrieval(self) -> None:
        retrieval = _RecordingClient(
            "retrieval", "retrieval/fake", _resp(RETRIEVAL_HIT, "retrieval/fake")
        )
        fine_tune = _RecordingClient(
            "retrieval", "qlora-adapter:fake", _resp(FINE_TUNE_HIT, "qlora-adapter:fake")
        )
        router = HybridRouter(retrieval, fine_tune)
        out = router.route(GOLD, run_id="r", batch_id="b")

        assert out.provenance == "gold-retrieval"
        assert out.candidate is not None
        assert out.candidate.citation_id == "Section 24"
        # The retrieval mapping_justification was the one returned (not fine-tune)
        assert "Retrieved from gold pair" in out.candidate.mapping_justification
        # Fine-tune must NOT have been called — that's the whole point of routing
        assert len(fine_tune.calls) == 0
        assert len(retrieval.calls) == 1
        # Both responses preserved structurally, but fine_tune_response stays None
        assert out.retrieval_response.top1 == RETRIEVAL_HIT
        assert out.fine_tune_response is None

    def test_retrieval_miss_falls_through_to_fine_tune(self) -> None:
        retrieval = _RecordingClient(
            "retrieval",
            "retrieval/fake",
            _resp(None, "retrieval/fake", parse_error="no confident retrieval match (cosine=0.42)"),
        )
        fine_tune = _RecordingClient(
            "retrieval", "qlora-adapter:fake", _resp(FINE_TUNE_HIT, "qlora-adapter:fake")
        )
        router = HybridRouter(retrieval, fine_tune)
        out = router.route(GOLD, run_id="r", batch_id="b")

        assert out.provenance == "fine-tune-generalization"
        assert out.candidate is not None
        # Fine-tune's justification surfaced (not retrieval's)
        assert "general security obligation" in out.candidate.mapping_justification
        # Both clients called exactly once
        assert len(retrieval.calls) == 1
        assert len(fine_tune.calls) == 1
        # Both responses surfaced
        assert out.retrieval_response.parse_error is not None
        assert out.fine_tune_response is not None
        assert out.fine_tune_response.top1 == FINE_TUNE_HIT

    def test_both_miss_returns_no_confident_match(self) -> None:
        retrieval = _RecordingClient(
            "retrieval",
            "retrieval/fake",
            _resp(None, "retrieval/fake", parse_error="no confident retrieval match"),
        )
        fine_tune = _RecordingClient(
            "retrieval",
            "qlora-adapter:fake",
            _resp(None, "qlora-adapter:fake", parse_error="generator parse failure"),
        )
        router = HybridRouter(retrieval, fine_tune)
        out = router.route(GOLD, run_id="r", batch_id="b")

        assert out.provenance == "no-confident-match"
        assert out.candidate is None
        # Both responses surfaced for the consumer to render
        assert out.retrieval_response.parse_error is not None
        assert out.fine_tune_response is not None
        assert out.fine_tune_response.parse_error is not None

    def test_route_passes_source_clause_and_target_jurisdiction(self) -> None:
        """Regression guard — both PromptMessages-extension fields populated.

        RetrievalClient depends on `source_clause_text` + `target_jurisdiction`
        being populated. If route() ever stops going through build_eval_prompt,
        retrieval would silently return parse_error="caller did not populate".
        """
        retrieval = _RecordingClient(
            "retrieval", "retrieval/fake", _resp(RETRIEVAL_HIT, "retrieval/fake")
        )
        fine_tune = _RecordingClient(
            "retrieval", "qlora-adapter:fake", _resp(FINE_TUNE_HIT, "qlora-adapter:fake")
        )
        router = HybridRouter(retrieval, fine_tune)
        router.route(GOLD, run_id="r", batch_id="b")

        assert len(retrieval.calls) == 1
        sent = retrieval.calls[0]
        assert sent.source_clause_text == "Security of processing."
        assert sent.target_jurisdiction == "sg"
        # The system + user prompt parts are also populated (full PromptMessages)
        assert sent.system  # non-empty
        assert "Art. 32" in sent.user
