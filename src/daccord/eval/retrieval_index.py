"""Retrieval-baseline index — FAISS over gold-pair source clauses.

Tier-12B deliverable. Builds an embedding index from a `GoldSet` (typically
the train split) so the retrieval baseline (and, at serving time, the
hybrid router in `daccord.serving.hybrid`) can answer queries by
nearest-neighbor lookup against validated gold mappings.

Two on-disk artifacts per index:
  - `<output_stem>.faiss`  — the FAISS index (`IndexFlatIP` over L2-
    normalized embeddings, so inner-product == cosine similarity).
  - `<output_stem>.jsonl`  — parallel rows of `RetrievalIndexEntry`, one
    per index row. Same order as the FAISS vectors — row `i` in the
    FAISS index corresponds to JSONL line `i`.

The two-file split keeps the FAISS binary opaque (re-buildable) while the
JSONL is git-diff-friendly + inspectable. The combined hash of both files
goes into `dataset_hash` on consumers so a stale index is detectable.

Heavy deps (`faiss`, `sentence_transformers`) are deferred-imported inside
each function so the shared `daccord` package stays importable in envs
that don't carry them (e.g. the root env at `pyproject.toml`). They live
in `envs/eval/pyproject.toml` and `consumer/pyproject.toml`.
"""

from __future__ import annotations

from pathlib import Path

from daccord.gold import GoldSet
from daccord.validation import ValidatedModel, validated


class RetrievalIndexEntry(ValidatedModel):
    """One row in the parallel JSONL — everything the retrieval client needs
    to construct a `ModelResponse` for a top-1 hit without re-reading the
    original gold file.

    `gold_id`, `target_citation_id`, `target_mechanism`, and
    `target_jurisdiction` are copied straight from the source `GoldPair`.
    `source_clause_text` is the string that was embedded — kept here so a
    consumer can show "you queried X, the nearest indexed clause was Y"
    when surfacing retrieval provenance.
    """

    gold_id: str
    source_clause_text: str
    target_jurisdiction: str
    target_framework: str
    target_citation_id: str
    target_mechanism: str


class TargetClauseIndexEntry(ValidatedModel):
    """One row in a target-clause index (tier-6B++ RAG seat input).

    Distinct from `RetrievalIndexEntry` because we're indexing target-framework
    clauses keyed by citation_id (e.g. `gdpr.json` → `clauses["32"]` text),
    not gold-pair source clauses keyed by gold_id. The RAG seat (tier 6B++)
    queries these indices with source-clause text to find the most semantically
    similar target citation; the result becomes a 5th "vote" in tier 7A.
    """

    framework: str
    citation_id: str  # normalised (matches the format tier 6B expects)
    citation_id_raw: str  # original key as it appears in clauses.json (pre-normalisation)
    clause_text: str


def _stem_paths(output_path: Path) -> tuple[Path, Path]:
    """Resolve `<stem>.faiss` + `<stem>.jsonl` from a single argument.

    Accepts either `foo` (no suffix), `foo.faiss`, or `foo.jsonl` — strips
    a `.faiss`/`.jsonl` suffix if present and reuses the stem for both
    files. Keeps the CLI surface forgiving without coupling callers to
    one suffix.
    """
    suffix = output_path.suffix.lower()
    stem = output_path.with_suffix("") if suffix in (".faiss", ".jsonl") else output_path
    return stem.with_suffix(".faiss"), stem.with_suffix(".jsonl")


@validated
def build_index(gold: GoldSet, embedder_name: str, output_path: Path) -> tuple[Path, Path]:
    """Embed every gold pair's `source_mechanism` and write a FAISS index.

    Returns `(faiss_path, jsonl_path)` — the two artifacts written. Caller
    typically passes a stem like `data/indices/retrieval__train__<hash>`;
    the suffixes are derived.

    Empty `gold` raises rather than writing a zero-row index — a silent
    empty index would surface much later as 100%-miss eval results.
    """
    if not gold.pairs:
        raise ValueError(
            f"refusing to build empty retrieval index from {gold.source_path!r}: "
            "no gold pairs to embed"
        )

    try:
        import faiss  # type: ignore[import-not-found]
        import numpy as np
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover — deps live in envs/eval + consumer
        raise RuntimeError(
            "retrieval index build requires faiss-cpu + sentence-transformers + numpy "
            "(install in envs/eval or consumer env)"
        ) from exc

    faiss_path, jsonl_path = _stem_paths(output_path)
    faiss_path.parent.mkdir(parents=True, exist_ok=True)

    entries = [
        RetrievalIndexEntry(
            gold_id=p.id,
            source_clause_text=p.source_mechanism,
            target_jurisdiction=p.target_jurisdiction,
            target_framework=p.target_framework,
            target_citation_id=p.target_citation_id,
            target_mechanism=p.target_mechanism,
        )
        for p in gold.pairs
    ]

    embedder = SentenceTransformer(embedder_name)
    # L2-normalize so IndexFlatIP returns cosine similarity in [-1, 1].
    raw_vecs = embedder.encode(
        [e.source_clause_text for e in entries],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    vecs = np.ascontiguousarray(raw_vecs, dtype=np.float32)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(faiss_path))

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(e.model_dump_json() + "\n")

    return faiss_path, jsonl_path


@validated
def build_target_clause_index(
    framework: str,
    clauses: dict[str, str],
    embedder_name: str,
    output_path: Path,
) -> tuple[Path, Path]:
    """Embed every (citation_id, clause_text) pair and write a per-framework FAISS index.

    Used by tier 6B++ (RAG seat): one index per target framework, queried with
    source-clause text from any framework. Empty-text clauses are skipped
    (tier-5's `body_recall` < 1.0 means a handful of registry IDs lack
    extracted body; an empty embedding would be noise in the similarity
    search).

    Returns `(faiss_path, jsonl_path)`. Caller passes a stem like
    `data/indices/target_clauses/gdpr` — suffixes derived.

    Citation IDs are stored both in their original form (as the dict key
    appears in `clauses.json`) and in normalised form (matching what tier 6B's
    fuzzy classifier uses), so the RAG seat can emit the normalised form
    without re-running `normalize_citation_id` per query.
    """
    from daccord.eval.scoring import normalize_citation_id

    if not clauses:
        raise ValueError(
            f"refusing to build empty target-clause index for framework={framework!r}: "
            "no clauses to embed"
        )

    try:
        import faiss  # type: ignore[import-not-found]
        import numpy as np
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover — deps live in envs/eval + consumer
        raise RuntimeError(
            "target-clause index build requires faiss-cpu + sentence-transformers + numpy "
            "(install in envs/eval or consumer env)"
        ) from exc

    faiss_path, jsonl_path = _stem_paths(output_path)
    faiss_path.parent.mkdir(parents=True, exist_ok=True)

    entries = [
        TargetClauseIndexEntry(
            framework=framework,
            citation_id=normalize_citation_id(raw_id),
            citation_id_raw=raw_id,
            clause_text=text,
        )
        for raw_id, text in clauses.items()
        if text and text.strip()
    ]
    if not entries:
        raise ValueError(
            f"refusing to build empty target-clause index for framework={framework!r}: "
            f"all {len(clauses)} clauses had empty text"
        )

    embedder = SentenceTransformer(embedder_name)
    raw_vecs = embedder.encode(
        [e.clause_text for e in entries],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    vecs = np.ascontiguousarray(raw_vecs, dtype=np.float32)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(faiss_path))

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(e.model_dump_json() + "\n")

    return faiss_path, jsonl_path


def load_target_clause_index(index_path: Path) -> tuple[object, list[TargetClauseIndexEntry]]:
    """Load a previously-built target-clause index. Returns `(faiss_index, entries)`.

    Symmetric with `load_index()` but for the target-clause shape.
    """
    try:
        import faiss  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "target-clause index load requires faiss-cpu (install in envs/eval or consumer env)"
        ) from exc

    faiss_path, jsonl_path = _stem_paths(index_path)
    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
    if not jsonl_path.exists():
        raise FileNotFoundError(f"index sidecar JSONL not found: {jsonl_path}")

    index = faiss.read_index(str(faiss_path))
    entries: list[TargetClauseIndexEntry] = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                entries.append(TargetClauseIndexEntry.model_validate_json(line))
            except Exception as exc:
                raise ValueError(
                    f"{jsonl_path}:{lineno}: invalid TargetClauseIndexEntry: {exc}"
                ) from exc

    if index.ntotal != len(entries):
        raise ValueError(
            f"index/sidecar size mismatch: FAISS has {index.ntotal} vectors, "
            f"JSONL has {len(entries)} entries"
        )
    return index, entries


def load_index(index_path: Path) -> tuple[object, list[RetrievalIndexEntry]]:
    """Load a previously-built index. Returns `(faiss_index, entries)`.

    `faiss_index` is typed `object` because the `faiss` module is a
    deferred import — annotating it as `faiss.Index` here would force a
    top-level import. Callers that need the precise type can re-import
    `faiss` themselves.
    """
    try:
        import faiss  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "retrieval index load requires faiss-cpu (install in envs/eval or consumer env)"
        ) from exc

    faiss_path, jsonl_path = _stem_paths(index_path)
    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
    if not jsonl_path.exists():
        raise FileNotFoundError(f"index sidecar JSONL not found: {jsonl_path}")

    index = faiss.read_index(str(faiss_path))
    entries: list[RetrievalIndexEntry] = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                entries.append(RetrievalIndexEntry.model_validate_json(line))
            except Exception as exc:
                raise ValueError(
                    f"{jsonl_path}:{lineno}: invalid RetrievalIndexEntry: {exc}"
                ) from exc

    if index.ntotal != len(entries):
        raise ValueError(
            f"index/sidecar size mismatch: FAISS has {index.ntotal} vectors, "
            f"JSONL has {len(entries)} entries"
        )
    return index, entries
