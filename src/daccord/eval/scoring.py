"""Scoring layer for the eval harness.

Three concerns, kept separate for practical SoC:

  - **Tier 1 — citation exact match** (`normalize_citation_id`,
    `citation_match_top1`). Deterministic, cheap, no API. The
    normalization table is locked at M0 — any edit invalidates
    baseline comparability with M4.

  - **Tier 2 — LLM-as-judge semantic match** (`JudgeClient` protocol,
    `judge_pair`, `judge_pairs`). Continuous 0-1 score plus a bucket
    and one-sentence reasoning. The reasoning column is the input to
    M4's Tier-3 human spot-check calibration.

  - **Aggregation** (`aggregate_rows`). Per-jurisdiction,
    per-language, and per-framework-pair breakdowns required by the
    development plan's metric contract.

The `GeminiJudge` adapter (free-tier judge default) lives here rather
than in clients.py because it's a *judge* — different protocol,
different prompt-renderer, different output shape — even though it
hits the same SDK as `GeminiClient`.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from typing import Literal, Protocol

from daccord.costs import preflight, record_call
from daccord.costs.config import Provider
from daccord.eval._rpm import api_throttle, gemini_retry_on_transient
from daccord.eval.prompts import build_judge_prompt
from daccord.eval.schema import ModelResponse, PromptMessages
from daccord.gold import GoldPair
from daccord.validation import ValidatedModel, validated

JudgeBucket = Literal[
    "wrong",
    "partial_wrong",
    "partial_right",
    "substantively_right",
    "exact",
]
JUDGE_BUCKETS: tuple[JudgeBucket, ...] = (
    "wrong",
    "partial_wrong",
    "partial_right",
    "substantively_right",
    "exact",
)

# Locked at M0. Any edit invalidates baseline comparability with M4.
# Longest alternatives FIRST — Python regex alternation is leftmost-first,
# so `art\.?` before `article` would eat "Art" out of "Article" and leave
# "icle 32" behind.
_PREFIX_STRIP = re.compile(
    r"^(article|section|paragraph|clause|art\.?|sec\.?|para\.?|cl\.?|§)\s*",
    re.IGNORECASE,
)


@validated
def normalize_citation_id(raw: str) -> str:
    """Canonicalize a citation identifier for Tier-1 exact-match scoring.

    Locked at M0. Strips "Art.", "Article", "§", "Section", etc.; collapses
    whitespace adjacent to parens (so "32(1)" == "32 (1)"); reconstructs
    parens when the model emits sub-clauses as space-separated tokens
    ("32 1 a" -> "32(1)(a)"); lowercases.

    Two paths after prefix-strip — only one runs:
      - input already contains "(" → just strip adjacent whitespace
      - input has multiple whitespace-separated parts → reconstruct as
        head + "("+sub+")" for each tail token. Groq/Gemini occasionally
        emit sub-clauses without parens in JSON output.
    """
    s = raw.strip()
    if not s:
        return ""
    s = _PREFIX_STRIP.sub("", s).strip()
    if "(" in s:
        s = re.sub(r"\s*\(\s*", "(", s)
        s = re.sub(r"\s*\)\s*", ")", s)
    else:
        parts = s.split()
        if len(parts) > 1:
            s = parts[0] + "".join(f"({p})" for p in parts[1:] if p)
    return s.lower()


@validated
def citation_match_top1(predicted: str, expected: str) -> bool:
    """Tier-1 metric: normalized exact match on top-1 prediction."""
    if not predicted or not expected:
        return False
    return normalize_citation_id(predicted) == normalize_citation_id(expected)


@validated
def citation_match_topk(predicted: list[str], expected: str) -> bool:
    """Tier-1 top-K metric. Defined now, unused at M0; wired at M4 if
    top-K candidates are added to `ModelResponse.candidates`.
    """
    if not predicted or not expected:
        return False
    return any(citation_match_top1(p, expected) for p in predicted)


class JudgeScore(ValidatedModel):
    score: float
    bucket: JudgeBucket
    reasoning: str
    judge_model: str
    parse_error: str | None = None


class JudgeClient(Protocol):
    provider: Provider
    model: str

    def judge(self, messages: PromptMessages, *, run_id: str, batch_id: str) -> JudgeScore: ...


_JUDGE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "bucket": {"type": "string", "enum": list(JUDGE_BUCKETS)},
        "reasoning": {"type": "string"},
    },
    "required": ["score", "bucket", "reasoning"],
}


def _parse_judge(raw_text: str, judge_model: str) -> JudgeScore:
    """Parse judge JSON to `JudgeScore`. Tolerant of bucket mismatches and
    out-of-range scores — clips to [0,1], maps unknown buckets to `wrong`,
    so a single bad response doesn't crash a run of hundreds of pairs.
    """
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return JudgeScore(
            score=0.0,
            bucket="wrong",
            reasoning="judge output did not parse as JSON",
            judge_model=judge_model,
            parse_error=f"json decode at char {exc.pos}: {exc.msg}",
        )
    if not isinstance(payload, dict):
        return JudgeScore(
            score=0.0,
            bucket="wrong",
            reasoning="judge output was not a JSON object",
            judge_model=judge_model,
            parse_error=f"expected JSON object, got {type(payload).__name__}",
        )
    score = float(payload.get("score", 0.0) or 0.0)
    if math.isnan(score):
        score = 0.0
    score = max(0.0, min(1.0, score))
    bucket_raw = str(payload.get("bucket", "wrong"))
    bucket: JudgeBucket = bucket_raw if bucket_raw in JUDGE_BUCKETS else "wrong"  # type: ignore[assignment]
    reasoning = str(payload.get("reasoning", "")) or "<no reasoning returned>"
    return JudgeScore(
        score=score,
        bucket=bucket,
        reasoning=reasoning,
        judge_model=judge_model,
    )


class GroqJudge:
    """Free-tier LLM-as-judge using Llama 4 Scout via Groq.

    Default model: `meta-llama/llama-4-scout-17b-16e-instruct` — strongest
    current-generation free-tier Llama via Groq (17B active × 16E MoE).
    Bumped 2026-05-25 from `llama-3.3-70b-versatile`. Trade-off: when `groq`
    is in the generator pool with the same model id, the judge is technically
    self-judging — a known M0 noise term on 20 pairs; document in the M4 run
    notes and swap to a non-Llama judge (e.g., DeepSeek V3) for the full eval.

    Groq has no native JSON-schema constraint — output discipline comes from
    the prompt's inline schema description + `response_format=json_object`.
    `_parse_judge` already tolerates malformed output (clips score, maps
    unknown bucket → "wrong"), so one bad response doesn't crash the run.
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
    def judge(self, messages: PromptMessages, *, run_id: str, batch_id: str) -> JudgeScore:
        from groq import APIError  # type: ignore[import-not-found]

        est_in = (len(messages.system) + len(messages.user)) // 4
        # Judges emit short JSON (score + bucket + 1-sentence reasoning), but
        # if a future judge is a "thinking" model (Qwen 3, etc.) the same
        # max-tokens-cutoff bug would silently zero its scores. Universal
        # generous ceiling — non-thinking judges still stop early.
        est_out = 2000
        preflight(self.provider, self.model, est_in, est_out)
        api_throttle(self.provider)

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
            # One bad judge call shouldn't kill the run — record as a zero-score
            # row with the error surfaced in `reasoning`.
            return JudgeScore(
                score=0.0,
                bucket="wrong",
                reasoning=f"judge API error: {type(exc).__name__}: {exc}",
                judge_model=self.model,
                parse_error=f"groq api error: {type(exc).__name__}: {exc}",
            )
        _ = (time.perf_counter() - t0) * 1000.0  # latency intentionally not recorded for judge

        raw_text = resp.choices[0].message.content or ""
        usage = resp.usage
        actual_in = int(usage.prompt_tokens) if usage else est_in
        actual_out = int(usage.completion_tokens) if usage else (len(raw_text) // 4)
        record_call(
            self.provider, self.model, actual_in, actual_out, run_id=run_id, batch_id=batch_id
        )
        return _parse_judge(raw_text, self.model)


class GeminiJudge:
    """LLM-as-judge using Gemini 3.1 Flash Lite — alternative to GroqJudge.

    Kept for the rare case where a non-Llama-based judge is desirable (e.g.,
    when M4 swaps Llama into the generator pool and needs a different judge
    family to avoid self-judging bias). Free-tier cap: 15 RPM / 500 RPD
    (older `gemini-2.5-flash` was dropped — daily cap was as low as 20 RPD
    on some accounts).
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
    def judge(self, messages: PromptMessages, *, run_id: str, batch_id: str) -> JudgeScore:
        from google.genai import types  # type: ignore[import-not-found]

        est_in = (len(messages.system) + len(messages.user)) // 4
        est_out = 200
        preflight(self.provider, self.model, est_in, est_out)
        api_throttle(self.provider)

        config = types.GenerateContentConfig(
            system_instruction=messages.system,
            temperature=0.0,
            max_output_tokens=est_out,
            response_mime_type="application/json",
            response_json_schema=_JUDGE_JSON_SCHEMA,
        )
        t0 = time.perf_counter()
        resp = gemini_retry_on_transient(
            lambda: self._client.models.generate_content(
                model=self.model,
                contents=messages.user,
                config=config,
            )
        )
        _ = (time.perf_counter() - t0) * 1000.0  # latency intentionally not recorded for judge

        raw_text = resp.text or ""
        usage = resp.usage_metadata
        actual_in = int(getattr(usage, "prompt_token_count", None) or est_in)
        actual_out = int(getattr(usage, "candidates_token_count", None) or (len(raw_text) // 4))
        record_call(
            self.provider, self.model, actual_in, actual_out, run_id=run_id, batch_id=batch_id
        )
        return _parse_judge(raw_text, self.model)


def judge_pair(
    gold: GoldPair, response: ModelResponse, judge: JudgeClient, *, run_id: str, batch_id: str
) -> JudgeScore:
    """Tier-2 single-pair judge. Sentinel `JudgeScore` returned when the
    generator failed to parse (no candidate to judge — automatic 0).
    """
    if response.top1 is None:
        return JudgeScore(
            score=0.0,
            bucket="wrong",
            reasoning=f"generator parse failure: {response.parse_error or '<unknown>'}",
            judge_model=judge.model,
        )
    msgs = build_judge_prompt(gold, response.top1)
    return judge.judge(msgs, run_id=run_id, batch_id=batch_id)


def judge_pairs(
    pairs: list[tuple[GoldPair, ModelResponse]],
    judge: JudgeClient,
    *,
    run_id: str,
    batch_id: str,
) -> list[JudgeScore]:
    """Batched signature for tier 12B / M4. At M0 simply loops single-call;
    M4 swaps to a true batched API call (10 pairs per request) without
    changing this signature.
    """
    return [
        judge_pair(gold, resp, judge, run_id=run_id, batch_id=f"{batch_id}::{gold.id}")
        for gold, resp in pairs
    ]


class EvalRow(ValidatedModel):
    """One row of the per-pair eval CSV. Column order is the wire contract."""

    gold_id: str
    model: str
    source_jurisdiction: str
    source_framework: str
    target_jurisdiction: str
    target_framework: str
    source_language: str
    target_language: str
    predicted_citation_id: str
    expected_citation_id: str
    citation_match: int  # 0 or 1
    judge_score: float
    judge_bucket: JudgeBucket
    judge_reasoning: str


@validated
def build_eval_row(gold: GoldPair, response: ModelResponse, judge_score: JudgeScore) -> EvalRow:
    predicted = response.top1.citation_id if response.top1 is not None else ""
    return EvalRow(
        gold_id=gold.id,
        model=response.model,
        source_jurisdiction=gold.source_jurisdiction,
        source_framework=gold.source_framework,
        target_jurisdiction=gold.target_jurisdiction,
        target_framework=gold.target_framework,
        source_language=gold.source_language,
        target_language=gold.target_language,
        predicted_citation_id=predicted,
        expected_citation_id=gold.target_citation_id,
        citation_match=int(citation_match_top1(predicted, gold.target_citation_id)),
        judge_score=judge_score.score,
        judge_bucket=judge_score.bucket,
        judge_reasoning=judge_score.reasoning,
    )


class AggregateBreakdown(ValidatedModel):
    """Aggregate metrics for one slice (one jurisdiction, one language, etc.)."""

    n: int
    tier1_citation_match: float
    tier2_judge_mean: float
    tier2_judge_pct_above_0_7: float


class EvalAggregates(ValidatedModel):
    """Full aggregation of an eval run, broken down for MLflow logging."""

    overall: AggregateBreakdown
    by_target_jurisdiction: dict[str, AggregateBreakdown]
    by_target_language: dict[str, AggregateBreakdown]
    by_framework_pair: dict[str, AggregateBreakdown]


def _agg(rows: list[EvalRow]) -> AggregateBreakdown:
    n = len(rows)
    if n == 0:
        return AggregateBreakdown(
            n=0, tier1_citation_match=0.0, tier2_judge_mean=0.0, tier2_judge_pct_above_0_7=0.0
        )
    tier1 = sum(r.citation_match for r in rows) / n
    judge_mean = sum(r.judge_score for r in rows) / n
    judge_high = sum(1 for r in rows if r.judge_score >= 0.7) / n
    return AggregateBreakdown(
        n=n,
        tier1_citation_match=tier1,
        tier2_judge_mean=judge_mean,
        tier2_judge_pct_above_0_7=judge_high,
    )


def _group_by(rows: list[EvalRow], key: str) -> dict[str, list[EvalRow]]:
    groups: dict[str, list[EvalRow]] = defaultdict(list)
    for r in rows:
        groups[getattr(r, key)].append(r)
    return dict(groups)


def _group_by_framework_pair(rows: list[EvalRow]) -> dict[str, list[EvalRow]]:
    groups: dict[str, list[EvalRow]] = defaultdict(list)
    for r in rows:
        groups[f"{r.source_framework}__{r.target_framework}"].append(r)
    return dict(groups)


@validated
def aggregate_rows(rows: list[EvalRow]) -> EvalAggregates:
    """Aggregate per-model. Caller groups by model first when multiple
    generators are evaluated under one parent run.
    """
    return EvalAggregates(
        overall=_agg(rows),
        by_target_jurisdiction={
            k: _agg(v) for k, v in _group_by(rows, "target_jurisdiction").items()
        },
        by_target_language={k: _agg(v) for k, v in _group_by(rows, "target_language").items()},
        by_framework_pair={k: _agg(v) for k, v in _group_by_framework_pair(rows).items()},
    )


@validated
def bucket_counts(rows: list[EvalRow]) -> dict[JudgeBucket, int]:
    """Quick sanity histogram for MLflow / CSV summary."""
    c = Counter(r.judge_bucket for r in rows)
    return {b: c.get(b, 0) for b in JUDGE_BUCKETS}
