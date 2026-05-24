"""Eval-harness inference-time data shapes.

These are tier-2B-specific shapes consumed by [clients.py] and [scoring.py]:

  - `CitationCandidate` — the model's per-pair prediction (one structured
    answer matching the eval prompt's JSON schema).
  - `ModelResponse` — provider-neutral wrapper around the SDK reply, with
    parse errors surfaced rather than raised.
  - `PromptMessages` — provider-neutral chat-style prompt.

The *gold* shapes (`GoldPair`, `GoldSet`) live in [daccord.gold.schema]
because they're pipeline-spanning (training, ensemble, freeze, eval all
consume them) — importing them from here would force a training script to
reach into the eval namespace for its own data.
"""

from __future__ import annotations

from daccord.validation import ValidatedModel


class CitationCandidate(ValidatedModel):
    """One predicted mapping. M0 emits a single candidate; top-K extension
    point preserved for tier 12B / M4 (defined here, unused at M0).
    """

    citation_id: str
    target_mechanism: str
    mapping_justification: str


class ModelResponse(ValidatedModel):
    """Normalized inference result across all `ModelClient` adapters.

    `candidates` is `None` at M0 (top-1 only via `top1`). When top-K lands
    at M4, populate `candidates` and the scoring layer's
    `citation_match_topk` reads from it.

    `parse_error` is set non-None when an adapter received output it could
    not parse into a `CitationCandidate` (e.g. local Qwen returning
    malformed JSON). The runner records this as a Tier-1 miss with the
    error surfaced in `judge_reasoning`.
    """

    model: str
    top1: CitationCandidate | None
    candidates: list[CitationCandidate] | None = None
    raw_text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    parse_error: str | None = None


class PromptMessages(ValidatedModel):
    """Provider-neutral chat-style prompt.

    Adapters translate to the SDK-native shape:
      - Anthropic: `system=...` kwarg + `messages=[{role, content}, ...]`
      - OpenAI: `messages=[{role: "system", ...}, {role: "user", ...}]`
      - LocalHF: rendered via the model's chat template.

    `source_clause_text` and `target_jurisdiction` are populated by
    `build_eval_prompt` (tier 2B) and consumed by `RetrievalClient`
    (tier 12B retrieval baseline) — the retrieval client embeds the
    clause text directly rather than the full rendered prompt (which
    would dilute embedding quality with template boilerplate). API
    adapters (GroqClient, GeminiClient) ignore these fields.
    """

    system: str
    user: str
    source_clause_text: str | None = None
    target_jurisdiction: str | None = None
