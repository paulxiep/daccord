"""Tier-6B++ RAG seat tests — strategy behaviour with an injected fake embedder + index.

Network-free: we build a tiny real FAISS index using a deterministic fake
SentenceTransformer (mirrors the `test_retrieval_client.py` + `test_target_index.py`
pattern), then drive the `RAGSeat` strategy against it. The seat is exercised
end-to-end including the per-call `append_candidate` writer and resume-by-source_id.

Verifies:
  - Top-1 above threshold → EnsembleCandidate with normalised citation_id.
  - Top-1 below threshold → empty-citation "no analog" vote (parse_error=None).
  - Missing target index → parse_error="missing_index_for_{framework}".
  - Resume by source_id skips already-emitted rows on re-invocation.
  - Embedding cache: same source_mechanism reused across pairs is embedded once.
  - Immutability guard from strategy.py refuses duplicate source_id.
  - Run accounting: parse_ok / parse_errors / resumed_from_disk are correct.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from daccord.ensemble.prompt import BatchPrompt
from daccord.ensemble.schema import EnsembleCandidate
from daccord.ensemble.strategies.rag_seat import RAGSeat
from daccord.ensemble.strategy import (
    ImmutabilityViolation,
    load_completed_source_ids,
)
from daccord.eval.retrieval_index import build_target_clause_index


def _read_jsonl(path: Path) -> list[EnsembleCandidate]:
    return [
        EnsembleCandidate.model_validate_json(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
    def __init__(self, name: str = EMBEDDER_NAME) -> None:
        self.name = name
        self.calls = 0

    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        import numpy as np

        self.calls += len(texts)
        return np.asarray([_stable_hash_embed(t) for t in texts], dtype=np.float32)


@pytest.fixture
def fake_st(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the real SentenceTransformer (used by `build_target_clause_index`)
    with our deterministic fake so building indices doesn't pull MPNet."""
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeSentenceTransformer)


def _prompt(
    source_id: str,
    source_mechanism: str,
    *,
    source_framework: str = "gdpr",
    target_framework: str = "pdpa_sg",
) -> BatchPrompt:
    return BatchPrompt(
        record_id=source_id,
        source_id=source_id,
        source_jurisdiction="eu" if source_framework == "gdpr" else "x",
        source_framework=source_framework,
        source_citation_id=source_id.rsplit("-", 1)[-1],
        source_mechanism=source_mechanism,
        target_jurisdiction="sg" if target_framework == "pdpa_sg" else "x",
        target_framework=target_framework,
        system="(unused by RAG seat)",
        user="(unused by RAG seat)",
        max_tokens=256,
    )


@pytest.fixture
def indices_dir(tmp_path: Path, fake_st: None) -> Path:
    """Build one tiny target-clause index for `pdpa_sg`."""
    out_dir = tmp_path / "indices"
    out_dir.mkdir()
    build_target_clause_index(
        framework="pdpa_sg",
        clauses={
            "13": "Consent required before collection of personal data.",
            "24": "Security arrangements to protect personal data.",
            "26D": "Notification of data breaches to commissioner.",
        },
        embedder_name=EMBEDDER_NAME,
        output_path=out_dir / "pdpa_sg",
    )
    return out_dir


def test_top1_above_threshold_emits_normalised_citation(tmp_path: Path, indices_dir: Path) -> None:
    """Querying with the exact indexed text → cosine = 1.0 → citation_id emitted."""
    seat = RAGSeat(
        indices_dir=indices_dir,
        embedder=_FakeSentenceTransformer(),
        threshold=0.5,
    )
    out_dir = tmp_path / "raw_local"
    prompts = [
        _prompt("gdpr-1", "Security arrangements to protect personal data."),
    ]
    results = seat.run_pair("gdpr__pdpa_sg", prompts, out_dir, smoke=False)

    rr = results["local-rag-mpnet"]
    assert rr.parse_ok == 1
    assert rr.parse_errors == 0
    # Verify the on-disk row
    written = _read_jsonl(Path(rr.output_path))
    assert len(written) == 1
    cand = written[0]
    assert cand.citation_id == "24"
    assert cand.target_mechanism.startswith("Security arrangements")
    assert "Retrieval (cosine" in cand.mapping_justification
    assert cand.parse_error is None


def test_top1_below_threshold_emits_empty_citation(tmp_path: Path, indices_dir: Path) -> None:
    """Below-threshold top-1 → empty citation_id, parse_error stays None.

    The fuzzy classifier treats this as a legitimate 'no analog' vote
    (same as an LLM that returns ""), NOT as a missing vote (parse_error).
    """
    seat = RAGSeat(
        indices_dir=indices_dir,
        embedder=_FakeSentenceTransformer(),
        # 0.99 is unreachable for unrelated text under the hash-based fake
        # embedder (random unit vectors have cosine ~0); forces the
        # below-threshold path deterministically.
        threshold=0.99,
    )
    out_dir = tmp_path / "raw_local"
    results = seat.run_pair(
        "gdpr__pdpa_sg",
        [_prompt("gdpr-1", "Some unrelated text")],
        out_dir,
        smoke=False,
    )

    rr = results["local-rag-mpnet"]
    assert rr.parse_ok == 1
    assert rr.parse_errors == 0
    cand = _read_jsonl(Path(rr.output_path))[0]
    assert cand.citation_id == ""
    assert cand.parse_error is None
    assert "below threshold" in cand.mapping_justification


def test_missing_index_emits_parse_error(tmp_path: Path, indices_dir: Path) -> None:
    """Target framework has no index → parse_error row.

    Tier 6B treats parse_error rows as missing votes (different from a
    legitimate empty-citation no-analog vote).
    """
    seat = RAGSeat(
        indices_dir=indices_dir,
        embedder=_FakeSentenceTransformer(),
        threshold=0.5,
    )
    out_dir = tmp_path / "raw_local"
    # bdsg has no index in this fixture
    prompts = [_prompt("gdpr-1", "anything", target_framework="bdsg")]
    results = seat.run_pair("gdpr__bdsg", prompts, out_dir, smoke=False)

    rr = results["local-rag-mpnet"]
    assert rr.parse_errors == 1
    cand = _read_jsonl(Path(rr.output_path))[0]
    assert cand.citation_id == ""
    assert cand.parse_error == "missing_index_for_bdsg"


def test_resume_by_source_id_skips_completed(tmp_path: Path, indices_dir: Path) -> None:
    """First run writes rows; second run sees them on disk and skips."""
    out_dir = tmp_path / "raw_local"
    seat = RAGSeat(
        indices_dir=indices_dir,
        embedder=_FakeSentenceTransformer(),
        threshold=0.0,
    )
    prompts = [
        _prompt("gdpr-1", "Security arrangements to protect personal data."),
        _prompt("gdpr-2", "Consent required before collection of personal data."),
    ]

    rr1 = seat.run_pair("gdpr__pdpa_sg", prompts, out_dir, smoke=False)["local-rag-mpnet"]
    assert rr1.parse_ok == 2
    assert rr1.resumed_from_disk == 0

    # Re-run: both source_ids already on disk; nothing new processed.
    seat2 = RAGSeat(
        indices_dir=indices_dir,
        embedder=_FakeSentenceTransformer(),
        threshold=0.0,
    )
    rr2 = seat2.run_pair("gdpr__pdpa_sg", prompts, out_dir, smoke=False)["local-rag-mpnet"]
    assert rr2.total_processed == 0
    assert rr2.resumed_from_disk == 2


def test_embedding_cache_dedupes_within_run(tmp_path: Path, indices_dir: Path) -> None:
    """Same source_id repeated → embedder called once (caching), not N times."""
    fake = _FakeSentenceTransformer()
    seat = RAGSeat(indices_dir=indices_dir, embedder=fake, threshold=0.0)
    out_dir = tmp_path / "raw_local"

    # Three distinct source_ids → 3 unique embeddings.
    prompts = [
        _prompt("gdpr-1", "alpha source text"),
        _prompt("gdpr-2", "beta source text"),
        _prompt("gdpr-3", "gamma source text"),
    ]
    seat.run_pair("gdpr__pdpa_sg", prompts, out_dir, smoke=False)
    assert fake.calls == 3

    # Repeat one of them (different pair would also hit cache, but
    # immutability would refuse re-writing to the same file; use a
    # different pair to exercise cache without conflict).
    out_dir2 = tmp_path / "raw_local2"
    seat.run_pair(
        "gdpr__pdpa_my",
        [_prompt("gdpr-1", "alpha source text", target_framework="pdpa_sg")],
        out_dir2,
        smoke=False,
    )
    # gdpr-1 already embedded; no new embedder call.
    assert fake.calls == 3


def test_immutability_refuses_duplicate_source_id(tmp_path: Path, indices_dir: Path) -> None:
    """Manually re-call the seat on a source_id already on disk → ImmutabilityViolation.

    (Not the normal path — resume-by-source_id avoids this — but the
    underlying append_candidate guard is what enforces the immutability
    contract. Pin it here so a future refactor doesn't lose the guarantee.)
    """
    out_dir = tmp_path / "raw_local"
    seat = RAGSeat(
        indices_dir=indices_dir,
        embedder=_FakeSentenceTransformer(),
        threshold=0.0,
    )
    prompt = _prompt("gdpr-1", "Security arrangements to protect personal data.")
    seat.run_pair("gdpr__pdpa_sg", [prompt], out_dir, smoke=False)
    # Bypass resume-by-source_id by calling _compute_one_prompt + raw append directly.
    from daccord.ensemble.strategy import append_candidate, output_path_for

    out_path = output_path_for(out_dir, "gdpr__pdpa_sg", "local-rag-mpnet")
    cand = seat._compute_one_prompt(prompt)  # type: ignore[attr-defined]
    with pytest.raises(ImmutabilityViolation):
        append_candidate(out_path, cand)


def test_constructor_validates_top_k_and_threshold(tmp_path: Path) -> None:
    indices = tmp_path / "indices"
    indices.mkdir()
    with pytest.raises(ValueError, match="top_k must be"):
        RAGSeat(indices_dir=indices, embedder=_FakeSentenceTransformer(), top_k=0)
    with pytest.raises(ValueError, match="threshold must be"):
        RAGSeat(indices_dir=indices, embedder=_FakeSentenceTransformer(), threshold=1.5)


def test_models_property_returns_slug(tmp_path: Path) -> None:
    indices = tmp_path / "indices"
    indices.mkdir()
    seat = RAGSeat(indices_dir=indices, embedder=_FakeSentenceTransformer())
    assert seat.models == ["local-rag-mpnet"]


def test_completed_set_loaded_from_seat_output(tmp_path: Path, indices_dir: Path) -> None:
    """Round-trip: after run, load_completed_source_ids sees the rows."""
    out_dir = tmp_path / "raw_local"
    seat = RAGSeat(
        indices_dir=indices_dir,
        embedder=_FakeSentenceTransformer(),
        threshold=0.0,
    )
    prompts = [_prompt(f"gdpr-{i}", f"text {i}") for i in range(3)]
    seat.run_pair("gdpr__pdpa_sg", prompts, out_dir, smoke=False)

    from daccord.ensemble.strategy import output_path_for

    out_path = output_path_for(out_dir, "gdpr__pdpa_sg", "local-rag-mpnet")
    assert load_completed_source_ids(out_path) == {"gdpr-0", "gdpr-1", "gdpr-2"}
