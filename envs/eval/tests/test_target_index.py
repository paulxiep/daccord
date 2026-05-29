"""Tier-6B++ target-clause index tests — build round-trip + edge cases.

Network-free: monkey-patches `SentenceTransformer` with a deterministic
fake embedder (same hash-based pattern used in `test_retrieval_client.py`)
so tests don't pull the 470MB MPNet model.

Verifies:
  1. `build_target_clause_index` writes a usable `.faiss` + `.jsonl` pair.
  2. Empty-text clauses are skipped (don't appear in the index).
  3. Citation IDs are normalised on write (so RAG seat queries match
     tier 6B's keyspace).
  4. `load_target_clause_index` round-trip preserves all entries + order.
  5. Top-1 self-retrieval returns the right citation_id for a known input.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from daccord.eval.retrieval_index import (
    TargetClauseIndexEntry,
    build_target_clause_index,
    load_target_clause_index,
)

EMBEDDING_DIM = 16
EMBEDDER_NAME = "fake-embedder-for-tests"


def _stable_hash_embed(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    chunks: list[int] = []
    for i in range(EMBEDDING_DIM):
        h = hashlib.sha256(digest + i.to_bytes(2, "little")).digest()
        chunks.append(int.from_bytes(h[:4], "little"))
    raw = [(c / 2**31) - 1.0 for c in chunks]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]


class _FakeSentenceTransformer:
    def __init__(self, name: str) -> None:
        self.name = name

    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        import numpy as np

        return np.asarray([_stable_hash_embed(t) for t in texts], dtype=np.float32)


@pytest.fixture
def fake_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeSentenceTransformer)


def test_build_and_load_round_trip(tmp_path: Path, fake_embedder: None) -> None:
    clauses = {
        "1": "Subject matter and objectives.",
        "32": "Security of processing.",
        "5A": "Lawfulness of processing — special rule.",
    }
    out_stem = tmp_path / "gdpr"
    faiss_path, jsonl_path = build_target_clause_index(
        framework="gdpr",
        clauses=clauses,
        embedder_name=EMBEDDER_NAME,
        output_path=out_stem,
    )
    assert faiss_path.exists() and faiss_path.suffix == ".faiss"
    assert jsonl_path.exists() and jsonl_path.suffix == ".jsonl"

    index, entries = load_target_clause_index(out_stem)
    assert index.ntotal == 3
    assert len(entries) == 3
    by_raw = {e.citation_id_raw: e for e in entries}
    assert set(by_raw.keys()) == {"1", "32", "5A"}
    assert by_raw["32"].framework == "gdpr"
    assert by_raw["32"].clause_text == "Security of processing."


def test_empty_text_clauses_are_skipped(tmp_path: Path, fake_embedder: None) -> None:
    clauses = {
        "1": "Subject matter.",
        "2": "",  # empty string — body_recall gap
        "3": "   \n  ",  # whitespace-only — also a gap
        "32": "Security of processing.",
    }
    out_stem = tmp_path / "gdpr"
    build_target_clause_index(
        framework="gdpr",
        clauses=clauses,
        embedder_name=EMBEDDER_NAME,
        output_path=out_stem,
    )
    _, entries = load_target_clause_index(out_stem)
    assert {e.citation_id_raw for e in entries} == {"1", "32"}


def test_citation_ids_are_normalised_on_write(tmp_path: Path, fake_embedder: None) -> None:
    # Tier 6B's classifier uses normalised IDs ("Section 32" → "32"); the
    # index must store them in the same form so RAG seat queries match.
    clauses = {
        "Section 32": "Security of processing.",
        "Article 1": "Subject matter.",
    }
    out_stem = tmp_path / "gdpr"
    build_target_clause_index(
        framework="gdpr",
        clauses=clauses,
        embedder_name=EMBEDDER_NAME,
        output_path=out_stem,
    )
    _, entries = load_target_clause_index(out_stem)
    by_norm = {e.citation_id: e for e in entries}
    assert "32" in by_norm
    assert "1" in by_norm
    # citation_id_raw preserves the original key for audit / display
    assert by_norm["32"].citation_id_raw == "Section 32"
    assert by_norm["1"].citation_id_raw == "Article 1"


def test_top1_self_retrieval_returns_indexed_citation(tmp_path: Path, fake_embedder: None) -> None:
    """Embedding the same string twice cosine-matches itself at score 1.0."""
    import numpy as np

    clauses = {
        "1": "Alpha clause body text",
        "2": "Beta clause body text",
        "3": "Gamma clause body text",
    }
    out_stem = tmp_path / "gdpr"
    build_target_clause_index(
        framework="gdpr",
        clauses=clauses,
        embedder_name=EMBEDDER_NAME,
        output_path=out_stem,
    )
    index, entries = load_target_clause_index(out_stem)

    embedder = _FakeSentenceTransformer(EMBEDDER_NAME)
    query_vec = np.ascontiguousarray(embedder.encode(["Beta clause body text"]), dtype=np.float32)
    scores, idxs = index.search(query_vec, 1)
    top_entry = entries[int(idxs[0][0])]
    assert top_entry.citation_id == "2"
    assert float(scores[0][0]) == pytest.approx(1.0, abs=1e-5)


def test_empty_input_raises(tmp_path: Path, fake_embedder: None) -> None:
    with pytest.raises(ValueError, match="no clauses to embed"):
        build_target_clause_index(
            framework="gdpr",
            clauses={},
            embedder_name=EMBEDDER_NAME,
            output_path=tmp_path / "gdpr",
        )


def test_all_empty_input_raises(tmp_path: Path, fake_embedder: None) -> None:
    with pytest.raises(ValueError, match="all .* clauses had empty text"):
        build_target_clause_index(
            framework="gdpr",
            clauses={"1": "", "2": "   "},
            embedder_name=EMBEDDER_NAME,
            output_path=tmp_path / "gdpr",
        )


def test_entry_schema_roundtrip() -> None:
    """Pin the TargetClauseIndexEntry schema independently of FAISS."""
    e = TargetClauseIndexEntry(
        framework="gdpr",
        citation_id="32",
        citation_id_raw="Section 32",
        clause_text="Security of processing.",
    )
    restored = TargetClauseIndexEntry.model_validate_json(e.model_dump_json())
    assert restored == e
