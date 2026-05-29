"""Prompt builders for tier 2B (eval) and tier 6A (ensemble generation).

Prompts are the most-edited variable in this project. They live in their own
module so git-blame on a prompt change is independent of SDK glue edits.

Three sibling builders:
  - `build_eval_prompt(GoldPair)` — used by the M0/M4 eval harness. Single-
    answer JSON, unconstrained citations (no registry pin at M0).
  - `build_ensemble_prompt(...)` — tier 6A. Same task shape as the eval
    prompt, but the citation_id is registry-pinned: the model must pick a
    string from the target framework's registered citation_ids (loaded from
    `data/registry/{framework}.json`) or return empty.
  - `build_judge_prompt(GoldPair, candidate)` — LLM-as-judge Tier-2 rubric
    prompt; separate from the generator so the judge rubric can evolve
    independently.

The output schema for generators is fixed at M0:

    eval generator     → CitationCandidate (unconstrained)
    ensemble generator → CitationCandidate (citation_id ∈ target registry)
    judge              → JudgeScore (see scoring.py)

Both shapes are produced via SDK-native JSON-schema (Gemini structured-output
with Pydantic, Groq's `response_format={"type":"json_object"}` plus inline
schema instructions). The local HF client falls back to prompt-and-parse.
"""

from __future__ import annotations

from daccord.eval.schema import CitationCandidate, PromptMessages
from daccord.gold import GoldPair
from daccord.validation import validated

EVAL_SYSTEM = (
    "You are a regulatory compliance specialist mapping data-privacy controls "
    "across jurisdictions. Given a source clause from one privacy framework, "
    "return the analogous clause in the named target framework, with the "
    "exact citation identifier as it appears in the target framework, a "
    "one-to-two sentence summary of the target mechanism, and a one-to-two "
    "sentence justification for the mapping.\n\n"
    "Reply with a single JSON object matching this schema:\n"
    '  {"citation_id": str, "target_mechanism": str, "mapping_justification": str}\n\n'
    "If no analogous clause exists in the target framework, return an empty "
    "string for citation_id and explain why in mapping_justification."
)

EVAL_USER_TEMPLATE = (
    "Source jurisdiction: {source_jurisdiction}\n"
    "Source framework: {source_framework}\n"
    "Source citation: {source_citation_id}\n"
    "Source clause: {source_mechanism}\n"
    "\n"
    "Target jurisdiction: {target_jurisdiction}\n"
    "Target framework: {target_framework}\n"
    "\n"
    "Return the analogous target clause as a JSON object."
)

JUDGE_SYSTEM = (
    "You are evaluating whether a proposed cross-jurisdiction regulatory "
    "mapping is substantively correct. The gold answer was produced by a "
    "human reviewer; the candidate was produced by an LLM under evaluation. "
    "Score whether the candidate captures the same substantive requirement "
    "as the gold answer.\n\n"
    "Reply with a single JSON object matching this schema:\n"
    '  {"score": float between 0.0 and 1.0,\n'
    '   "bucket": one of "wrong", "partial_wrong", "partial_right",\n'
    '             "substantively_right", "exact",\n'
    '   "reasoning": one-sentence explanation}\n\n'
    "Anchor the score: 0.0 = unrelated control; 0.25 = same domain but wrong "
    "mechanism; 0.5 = adjacent mechanism, missing key element; 0.75 = same "
    "mechanism, minor scope mismatch; 1.0 = substantively identical. Do not "
    "penalise valid paraphrasing or minor citation-format differences."
)

JUDGE_USER_TEMPLATE = (
    "Source clause ({source_framework} {source_citation_id}):\n"
    "  {source_mechanism}\n"
    "\n"
    "GOLD answer ({target_framework} {expected_citation_id}):\n"
    "  {expected_mechanism}\n"
    "\n"
    "CANDIDATE answer ({target_framework} {predicted_citation_id}):\n"
    "  {predicted_mechanism}\n"
    "  Mapping justification: {predicted_justification}\n"
    "\n"
    "Score the candidate."
)


@validated
def build_eval_prompt(gold_pair: GoldPair) -> PromptMessages:
    """Generator prompt for the M0/M4 eval harness — single-answer JSON.

    Citations are NOT registry-pinned at M0 (tag CSV with
    `prompt_variant='unconstrained-m0'`). When tier 5 lands the registry,
    `build_ensemble_prompt` is the registry-constrained sibling — same task
    shape, stricter citation discipline.
    """
    user = EVAL_USER_TEMPLATE.format(
        source_jurisdiction=gold_pair.source_jurisdiction,
        source_framework=gold_pair.source_framework,
        source_citation_id=gold_pair.source_citation_id,
        source_mechanism=gold_pair.source_mechanism,
        target_jurisdiction=gold_pair.target_jurisdiction,
        target_framework=gold_pair.target_framework,
    )
    return PromptMessages(
        system=EVAL_SYSTEM,
        user=user,
        # Surfaced as separate fields so RetrievalClient (tier 12B) can
        # embed the clause text directly + filter the index by target
        # jurisdiction, instead of trying to parse them back out of `user`.
        # API clients ignore these.
        source_clause_text=gold_pair.source_mechanism,
        target_jurisdiction=gold_pair.target_jurisdiction,
    )


ENSEMBLE_SYSTEM = (
    "You are a regulatory compliance specialist mapping data-privacy controls "
    "across jurisdictions. Given a source clause from one privacy framework, "
    "return the analogous clause in the named target framework, with the "
    "exact citation identifier as it appears in the target framework, a "
    "one-to-two sentence summary of the target mechanism, and a one-to-two "
    "sentence justification for the mapping.\n\n"
    "STRICT CONSTRAINT: The citation_id you return MUST be one of the exact "
    "string values listed under 'Valid citation_ids' in the user message. "
    "Do not invent citation IDs. Do not paraphrase them. Do not add prefixes "
    'like "Article" or "Section" — return the bare identifier as listed.\n\n'
    "Reply with a single JSON object matching this schema:\n"
    '  {"citation_id": str, "target_mechanism": str, "mapping_justification": str}\n\n'
    "If no analogous clause exists in the target framework, return an empty "
    "string for citation_id and explain why in mapping_justification."
)

ENSEMBLE_USER_TEMPLATE = (
    "Source jurisdiction: {source_jurisdiction}\n"
    "Source framework: {source_framework}\n"
    "Source citation: {source_citation_id}\n"
    "Source clause: {source_mechanism}\n"
    "\n"
    "Target jurisdiction: {target_jurisdiction}\n"
    "Target framework: {target_framework}\n"
    "Valid citation_ids for {target_framework} ({registry_count} total) — "
    "you MUST pick exactly one of these, or empty string if no analogous "
    "clause exists:\n"
    "{registry_block}\n"
    "\n"
    "Return the analogous target clause as a JSON object."
)


def _format_registry_block(citation_ids: list[str]) -> str:
    # Comma-separated quoted IDs. Compact (saves ~50% tokens vs bullet list)
    # while still clearly delimited so the model treats each as a literal
    # string. 288-entry DPA-2018 registry renders to ~1.5 KB / ~400 tokens.
    return ", ".join(f'"{cid}"' for cid in citation_ids)


@validated
def build_ensemble_prompt(
    *,
    source_jurisdiction: str,
    source_framework: str,
    source_citation_id: str,
    source_mechanism: str,
    target_jurisdiction: str,
    target_framework: str,
    target_registry: list[str],
) -> PromptMessages:
    """Tier 6A ensemble generator prompt — citation_id pinned to registry.

    Same task shape as `build_eval_prompt` but with two differences:
      1. `target_registry` is stuffed into the user message; the model must
         return a citation_id from this set (or empty string).
      2. There is no GoldPair — at tier 7A the gold answer doesn't exist
         yet; the ensemble's 4-way agreement on citation_id is what tier 8
         tiers HIGH/MED/LOW/SALVAGE from.

    Tag CSV / MLflow runs with `prompt_variant='ensemble-registry-pinned'`
    so a prompt change between tier 7A runs is detectable in the run
    history.

    Keyword-only args: 7 fields all of similar type — positional ordering
    is a foot-gun, especially the source/target string pairs.
    """
    user = ENSEMBLE_USER_TEMPLATE.format(
        source_jurisdiction=source_jurisdiction,
        source_framework=source_framework,
        source_citation_id=source_citation_id,
        source_mechanism=source_mechanism,
        target_jurisdiction=target_jurisdiction,
        target_framework=target_framework,
        registry_count=len(target_registry),
        registry_block=_format_registry_block(target_registry),
    )
    return PromptMessages(
        system=ENSEMBLE_SYSTEM,
        user=user,
        source_clause_text=source_mechanism,
        target_jurisdiction=target_jurisdiction,
    )


@validated
def build_judge_prompt(gold_pair: GoldPair, candidate: CitationCandidate) -> PromptMessages:
    """Tier-2 LLM-as-judge prompt: continuous 0-1 score + bucket + reasoning.

    The `reasoning` field is the seed for M4's human spot-check (tier 3) — a
    stratified sample of low-score rows is reviewed manually to calibrate
    the judge against ground truth.
    """
    user = JUDGE_USER_TEMPLATE.format(
        source_framework=gold_pair.source_framework,
        source_citation_id=gold_pair.source_citation_id,
        source_mechanism=gold_pair.source_mechanism,
        target_framework=gold_pair.target_framework,
        expected_citation_id=gold_pair.target_citation_id,
        expected_mechanism=gold_pair.target_mechanism,
        predicted_citation_id=candidate.citation_id or "<no citation returned>",
        predicted_mechanism=candidate.target_mechanism or "<no mechanism returned>",
        predicted_justification=candidate.mapping_justification or "<no justification returned>",
    )
    return PromptMessages(system=JUDGE_SYSTEM, user=user)
