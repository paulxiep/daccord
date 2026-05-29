"""Tier-6B++ — RAG seat (a 5th `EnsembleStrategy` peer to the 4 LLM seats).

Produces `EnsembleCandidate`-shape rows from semantic-similarity retrieval
against pre-built per-framework target-clause indices (output of
`envs/eval/scripts/build_target_indices.py`). The 5th seat slots into
tier-7A's ensemble naturally:

  - Output file naming: `{framework_pair}__local-rag-mpnet.jsonl`, matching
    the LLM-seat convention.
  - Storage convention: `data/ensemble/raw_local/` rather than
    `data/ensemble/raw/`. raw/ is the immutable paid-API record; raw_local/
    is for re-runnable local compute (RAG today; future local HF seats too).
    The same per-call fsync + `ImmutabilityViolation` guards from
    `daccord.ensemble.strategy` still apply within a run.
  - Tier 6B's `tier_framework_pair` globs both raw/ + raw_local/ so the
    fuzzy classifier picks up 5 votes per source clause automatically. The
    RAG vote contributes to `agreement_score` with equal weight to LLM
    votes — concurrence is informative; dissent legitimately lowers
    confidence.

The seat is a different epistemic angle from the LLM seats: it does no
reasoning, just finds the most semantically similar target clause text.
Failure modes are uncorrelated with the LLMs (which can hallucinate
citations not in the registry; retrieval can only return registry
citations, but may pick the wrong one when surface similarity diverges
from legal-semantic match).

## Cost + wall-clock

CPU-only embedding (paraphrase-multilingual-mpnet-base-v2) + FAISS lookup.
~5-10 min over all 72 pairs × ~95 source clauses average ≈ 6,800 embed +
lookup pairs. **~$0 API spend.** Embedding cache by `source_id` means each
source is embedded once per run even when it appears across 8 target pairs.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Protocol

from daccord.ensemble.prompt import BatchPrompt
from daccord.ensemble.schema import EnsembleCandidate
from daccord.ensemble.strategy import (
    RunResult,
    append_candidate,
    load_completed_source_ids,
    make_error_candidate,
    output_path_for,
)
from daccord.eval.retrieval_index import (
    TargetClauseIndexEntry,
    load_target_clause_index,
)
from daccord.validation import validated

log = logging.getLogger("daccord.ensemble.strategies.rag_seat")

DEFAULT_EMBEDDER = "paraphrase-multilingual-mpnet-base-v2"
DEFAULT_MODEL_SLUG = "local-rag-mpnet"
DEFAULT_THRESHOLD = 0.4
DEFAULT_TOP_K = 5


class _EmbedderProtocol(Protocol):
    """Subset of `SentenceTransformer` the seat actually calls.

    Defined as a Protocol so tests can inject a deterministic fake without
    importing torch / sentence-transformers in the root env.
    """

    def encode(self, texts: list[str], **kwargs: Any) -> Any: ...


def _load_default_embedder(embedder_name: str) -> _EmbedderProtocol:
    """Lazy-import + construct the real `SentenceTransformer`.

    Kept out of the class so tests can pass a fake embedder via
    `RAGSeat(embedder=fake)` without triggering the heavy import.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover — eval env carries the dep
        raise RuntimeError(
            "RAGSeat needs sentence-transformers (install in envs/eval or consumer env)"
        ) from exc
    return SentenceTransformer(embedder_name)


class RAGSeat:
    """A 5th-seat ensemble strategy backed by per-framework target-clause indices.

    Construct once per process; reuse across pairs to amortise the model
    load and the per-framework index loads. The embedding cache is keyed
    by `source_id` so repeat sources (each gdpr source clause appears in
    8 forward pairs) are embedded once.

    `name` + `models` mirror the `PaidAPIStrategy` shape so callers can
    treat any strategy uniformly. `models` returns a single-element list
    (`[DEFAULT_MODEL_SLUG]`) because there's only one "seat" — the
    retrieval index is a single producer of EnsembleCandidates.

    `run_pair` writes to `{out_dir}/{framework_pair}__{model_slug}.jsonl`.
    For the canonical raw_local/ layout, pass `out_dir=data/ensemble/raw_local`.
    """

    name = "local-rag"

    def __init__(
        self,
        *,
        indices_dir: Path,
        embedder: _EmbedderProtocol | None = None,
        embedder_name: str = DEFAULT_EMBEDDER,
        model_slug: str = DEFAULT_MODEL_SLUG,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        # No `@validated` on __init__ because pydantic.validate_call can't
        # build an isinstance check for the `_EmbedderProtocol` parameter
        # (Protocols aren't runtime-checkable without @runtime_checkable +
        # a real isinstance dispatch). The constructor isn't a boundary
        # anyway — caller-controlled construction with controlled types.
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self._indices_dir = indices_dir
        self._embedder = embedder if embedder is not None else _load_default_embedder(embedder_name)
        self._embedder_name = embedder_name
        self._model_slug = model_slug
        self._top_k = top_k
        self._threshold = threshold
        # Per-process caches keyed by framework / source_id so repeated
        # source clauses across the 8 target pairs are embedded once.
        self._index_cache: dict[str, tuple[object, list[TargetClauseIndexEntry]] | None] = {}
        self._embedding_cache: dict[str, Any] = {}

    @property
    def models(self) -> list[str]:
        return [self._model_slug]

    def run_pair(
        self,
        framework_pair: str,
        prompts: list[BatchPrompt],
        out_dir: Path,
        *,
        smoke: bool,
    ) -> dict[str, RunResult]:
        _ = smoke  # accepted to match the EnsembleStrategy protocol
        start = time.monotonic()
        out_path = output_path_for(out_dir, framework_pair, self._model_slug)

        completed = load_completed_source_ids(out_path)
        if completed:
            log.info(
                "[rag-seat] %s: resuming, %d source_ids already on disk",
                framework_pair,
                len(completed),
            )
        remaining = [p for p in prompts if p.source_id not in completed]

        parse_ok = 0
        parse_errors = 0
        for prompt in remaining:
            candidate = self._compute_one_prompt(prompt)
            try:
                append_candidate(out_path, candidate)
            except Exception as exc:
                log.error(
                    "[rag-seat] disk write failed for %s / %s: %s",
                    framework_pair,
                    prompt.source_id,
                    exc,
                )
                raise
            if candidate.parse_error is None:
                parse_ok += 1
            else:
                parse_errors += 1

        elapsed = time.monotonic() - start
        log.info(
            "[rag-seat] %s done: processed=%d ok=%d errors=%d resumed=%d (%.1fs)",
            framework_pair,
            len(remaining),
            parse_ok,
            parse_errors,
            len(completed),
            elapsed,
        )
        return {
            self._model_slug: RunResult(
                framework_pair=framework_pair,
                model=self._model_slug,
                output_path=str(out_path),
                total_processed=len(remaining),
                parse_ok=parse_ok,
                parse_errors=parse_errors,
                resumed_from_disk=len(completed),
                seconds_elapsed=elapsed,
            )
        }

    @validated
    def _compute_one_prompt(self, prompt: BatchPrompt) -> EnsembleCandidate:
        """Retrieve top-K target clauses for `prompt.source_mechanism`; emit one EnsembleCandidate.

        Failure modes captured as `parse_error`-flagged candidates (so tier
        6B treats them as missing votes, identical to an LLM that
        timed-out):
          - Missing target framework index → "missing_index_for_{framework}".
          - Below-threshold top-1 → empty `citation_id` + `parse_error=None`
            (treated as "no analog exists" by the fuzzy classifier — same
            as an LLM that legitimately returns "").

        Success: emits `citation_id` = normalised top-1, `target_mechanism`
        = retrieved clause text, `mapping_justification` records the
        retrieval provenance (cosine score + top-K snapshot for audit).
        """
        loaded = self._get_index(prompt.target_framework)
        if loaded is None:
            return make_error_candidate(
                prompt=prompt,
                model=self._model_slug,
                error_message=f"missing_index_for_{prompt.target_framework}",
            )

        index, entries = loaded
        try:
            scores, idxs = self._search(prompt.source_id, prompt.source_mechanism, index)
        except Exception as exc:
            return make_error_candidate(
                prompt=prompt,
                model=self._model_slug,
                error_message=f"{type(exc).__name__}: {exc}",
            )

        if len(idxs) == 0 or len(entries) == 0:
            return make_error_candidate(
                prompt=prompt,
                model=self._model_slug,
                error_message="empty_target_index",
            )

        top1_score = float(scores[0])
        top1_entry = entries[int(idxs[0])]
        top_k_summary = ", ".join(
            f"{entries[int(i)].citation_id}@{float(s):.3f}"
            for i, s in zip(idxs[: self._top_k], scores[: self._top_k], strict=False)
        )

        if top1_score < self._threshold:
            # Below threshold → treat as "no analog" (empty citation_id;
            # parse_error stays None so fuzzy classifier counts this as a
            # legitimate "no" vote, not a missing vote).
            return EnsembleCandidate(
                source_id=prompt.source_id,
                source_jurisdiction=prompt.source_jurisdiction,
                source_framework=prompt.source_framework,
                source_citation_id=prompt.source_citation_id,
                source_mechanism=prompt.source_mechanism,
                target_jurisdiction=prompt.target_jurisdiction,
                target_framework=prompt.target_framework,
                model=self._model_slug,
                citation_id="",
                target_mechanism="",
                mapping_justification=(
                    f"Retrieval below threshold {self._threshold} "
                    f"(top-1 score {top1_score:.3f}; top-{self._top_k}: {top_k_summary})"
                ),
                parse_error=None,
            )

        return EnsembleCandidate(
            source_id=prompt.source_id,
            source_jurisdiction=prompt.source_jurisdiction,
            source_framework=prompt.source_framework,
            source_citation_id=prompt.source_citation_id,
            source_mechanism=prompt.source_mechanism,
            target_jurisdiction=prompt.target_jurisdiction,
            target_framework=prompt.target_framework,
            model=self._model_slug,
            citation_id=top1_entry.citation_id,
            target_mechanism=top1_entry.clause_text,
            mapping_justification=(
                f"Retrieval (cosine {top1_score:.3f}; top-{self._top_k}: {top_k_summary})"
            ),
            parse_error=None,
        )

    def _get_index(self, framework: str) -> tuple[object, list[TargetClauseIndexEntry]] | None:
        """Load + cache one framework's index; None when the index file is missing."""
        if framework in self._index_cache:
            return self._index_cache[framework]
        index_stem = self._indices_dir / framework
        try:
            loaded = load_target_clause_index(index_stem)
        except FileNotFoundError:
            log.warning(
                "[rag-seat] no index for framework=%s at %s; emitting parse_error rows",
                framework,
                index_stem,
            )
            self._index_cache[framework] = None
            return None
        self._index_cache[framework] = loaded
        return loaded

    def _search(self, source_id: str, source_text: str, index: object) -> tuple[Any, Any]:
        """Embed `source_text` (cached by source_id) and search top-K on `index`.

        Returns `(scores_flat, idxs_flat)` — 1D arrays of length top_k.
        """
        import numpy as np

        if source_id in self._embedding_cache:
            query_vec = self._embedding_cache[source_id]
        else:
            raw_vec = self._embedder.encode(
                [source_text],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            query_vec = np.ascontiguousarray(raw_vec, dtype=np.float32)
            self._embedding_cache[source_id] = query_vec

        # FAISS `.search(queries, k)` returns (D, I) shaped (n_queries, k).
        scores, idxs = index.search(query_vec, self._top_k)  # type: ignore[attr-defined]
        return scores[0], idxs[0]
