"""Hybrid serving layer — retrieval-first + QLoRA-fallback with provenance tagging.

Wraps the eval-layer `ModelClient` Protocol (one for retrieval, one for the
fine-tuned QLoRA adapter) into a single `HybridRouter` whose output carries
a per-response `provenance` tag. Same router runs in two places:

  - **consumer/app.py** (local demo): Streamlit side-by-side comparison view
    invokes the router for the "fine-tune" column; the "retrieval" and
    "base" columns call those clients directly.
  - **publish/sagemaker_handler.py** (production endpoint): SageMaker
    custom inference handler wraps the router and emits the provenance
    in the response JSON so Pillar C's agent can branch on it (see
    aws_credential_plan.md Pillar C `cross_jurisdiction_map` tool).

The provenance contract is the load-bearing API surface:
  - `gold-retrieval`: top-1 cosine over the gold-pair index exceeded the
    threshold; answer is verbatim from a hand-validated gold pair.
  - `fine-tune-generalization`: retrieval missed the threshold; the
    QLoRA-fine-tuned model generated the answer.
  - `no-confident-match`: both retrieval and fine-tune failed (retrieval
    below threshold + fine-tune parse error or empty output).

Pillar C's `cross_jurisdiction_map` tool uses this signal to gate the
downstream `verify_citation` call: skip when gold-retrieval; trigger when
fine-tune-generalization; fall through to `search_regulation_text` when
no-confident-match.
"""

from daccord.serving.clients import FineTuneClient, LocalAdapterClient
from daccord.serving.hybrid import HybridResponse, HybridRouter, Provenance

__all__ = [
    "FineTuneClient",
    "HybridResponse",
    "HybridRouter",
    "LocalAdapterClient",
    "Provenance",
]
