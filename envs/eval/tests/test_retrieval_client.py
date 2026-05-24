"""Tier-12B retrieval-baseline tests — index build round-trip + client behavior.

Network-free: monkey-patches `SentenceTransformer` with a deterministic
fake embedder so tests don't pull the 470MB MPNet model. Validates four
contracts:

  1. `build_index` produces a usable `.faiss` + `.jsonl` pair.
  2. `RetrievalClient.generate()` returns a top-1 hit that matches the
     gold pair indexed under that source clause.
  3. Target-jurisdiction filtering kicks in — a query for jurisdiction X
     ignores indexed pairs targeting jurisdiction Y, even when their
     source clauses are textually closer.
  4. `score_threshold` produces a `top1=None` ModelResponse with the
     cosine surfaced in `parse_error` when the top-1 cosine is below the
     ceiling.

The fake embedder maps each input string to a fixed-dimension vector by
seeded hashing — deterministic, unit-norm, no dependency on real ML.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from daccord.eval.clients import RetrievalClient
from daccord.eval.retrieval_index import build_index, load_index
from daccord.eval.schema import PromptMessages
from daccord.gold import GoldPair, GoldSet

EMBEDDING_DIM = 16
EMBEDDER_NAME = "fake-embedder-for-tests"


def _stable_hash_embed(text: str) -> list[float]:
    """Deterministic, fixed-dim, L2-normalized vector from a string.

    Hash the input into 8 bytes per dimension, treat as little-endian int,
    map to [-1, 1] via /2^31 - 1, then L2-normalize. Two strings with the
    same hash collide perfectly; otherwise distinct. Good enough for
    nearest-neighbor unit tests without bringing in torch.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Use first 64 bytes; cycle if EMBEDDING_DIM*4 > 32. Stretch by
    # rehashing with index suffix.
    chunks: list[int] = []
    for i in range(EMBEDDING_DIM):
        h = hashlib.sha256(digest + i.to_bytes(2, "little")).digest()
        chunks.append(int.from_bytes(h[:4], "little"))
    raw = [(c / 2**31) - 1.0 for c in chunks]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]


class _FakeSentenceTransformer:
    """Stand-in for `sentence_transformers.SentenceTransformer`."""

    def __init__(self, name: str) -> None:
        self.name = name

    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        import numpy as np

        return np.asarray(
            [_stable_hash_embed(t) for t in texts], dtype=np.float32
        )


@pytest.fixture
def fake_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", _FakeSentenceTransformer
    )


def _mk_pair(
    pid: str,
    source_mech: str,
    target_jur: str,
    target_cit: str,
    target_mech: str,
) -> GoldPair:
    return GoldPair(
        id=pid,
        source_jurisdiction="eu",
        source_framework="gdpr",
        source_citation_id=f"Art. {pid}",
        source_mechanism=source_mech,
        source_language="en",
        target_jurisdiction=target_jur,
        target_framework=f"pdpa_{target_jur}",
        target_citation_id=target_cit,
        target_mechanism=target_mech,
        target_language="en",
        notes=None,
    )


@pytest.fixture
def tiny_gold(tmp_path: Path) -> GoldSet:
    pairs = [
        _mk_pair("p1", "Security of processing.", "sg", "Section 24",
                 "Reasonable security arrangements."),
        _mk_pair("p2", "Right to erasure.", "th", "Section 33",
                 "Right to be forgotten under Thai PDPA."),
        _mk_pair("p3", "Lawfulness, fairness, transparency.", "sg", "Section 13",
                 "Consent obligation."),
        # Same source text as p1 but target jurisdiction th — used to verify
        # jurisdiction filtering routes correctly even on collision.
        _mk_pair("p4", "Security of processing.", "th", "Section 37",
                 "Thai security obligation."),
    ]
    jsonl = tmp_path / "tiny.jsonl"
    jsonl.write_text(
        "\n".join(p.model_dump_json() for p in pairs) + "\n",
        encoding="utf-8",
    )
    return GoldSet.from_jsonl(jsonl)


class TestBuildIndex:
    def test_round_trip(
        self, tiny_gold: GoldSet, tmp_path: Path, fake_embedder: None
    ) -> None:
        out = tmp_path / "indices" / "tiny"
        faiss_path, jsonl_path = build_index(tiny_gold, EMBEDDER_NAME, out)
        assert faiss_path.exists()
        assert jsonl_path.exists()

        index, entries = load_index(out)
        assert index.ntotal == len(tiny_gold.pairs) == 4
        assert {e.gold_id for e in entries} == {"p1", "p2", "p3", "p4"}

    def test_empty_gold_rejected(self, tmp_path: Path, fake_embedder: None) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        gold = GoldSet.from_jsonl(empty)
        with pytest.raises(ValueError, match="empty retrieval index"):
            build_index(gold, EMBEDDER_NAME, tmp_path / "out")

    def test_accepts_stem_or_suffix(
        self, tiny_gold: GoldSet, tmp_path: Path, fake_embedder: None
    ) -> None:
        # Passing `.faiss` should not produce a `.faiss.faiss` file
        faiss_path, jsonl_path = build_index(
            tiny_gold, EMBEDDER_NAME, tmp_path / "tiny.faiss"
        )
        assert faiss_path.name == "tiny.faiss"
        assert jsonl_path.name == "tiny.jsonl"


class TestRetrievalClient:
    def test_top1_recovers_indexed_pair(
        self, tiny_gold: GoldSet, tmp_path: Path, fake_embedder: None
    ) -> None:
        out = tmp_path / "tiny"
        build_index(tiny_gold, EMBEDDER_NAME, out)
        client = RetrievalClient(index_path=out, embedder_name=EMBEDDER_NAME)
        # Query with the exact text of p1's source clause; expect p1 back.
        messages = PromptMessages(
            system="sys",
            user="usr",
            source_clause_text="Security of processing.",
            target_jurisdiction="sg",
        )
        resp = client.generate(messages, run_id="r", batch_id="b")
        assert resp.top1 is not None
        assert resp.top1.citation_id == "Section 24"
        assert resp.top1.target_mechanism == "Reasonable security arrangements."
        assert "p1" in resp.top1.mapping_justification
        # Provenance fields surfaced in raw_text JSON for the consumer
        import json

        raw = json.loads(resp.raw_text)
        assert raw["gold_id"] == "p1"
        assert "cosine" in raw

    def test_jurisdiction_filter_separates_colliding_source(
        self, tiny_gold: GoldSet, tmp_path: Path, fake_embedder: None
    ) -> None:
        # p1 and p4 share source text; differ only in target_jurisdiction.
        # The fake embedder collides them perfectly. Verify filter routes
        # each to its correct target.
        out = tmp_path / "tiny"
        build_index(tiny_gold, EMBEDDER_NAME, out)
        client = RetrievalClient(index_path=out, embedder_name=EMBEDDER_NAME)

        msg_sg = PromptMessages(
            system="s", user="u",
            source_clause_text="Security of processing.",
            target_jurisdiction="sg",
        )
        msg_th = PromptMessages(
            system="s", user="u",
            source_clause_text="Security of processing.",
            target_jurisdiction="th",
        )
        resp_sg = client.generate(msg_sg, run_id="r", batch_id="b")
        resp_th = client.generate(msg_th, run_id="r", batch_id="b")
        assert resp_sg.top1 is not None
        assert resp_sg.top1.citation_id == "Section 24"
        assert resp_th.top1 is not None
        assert resp_th.top1.citation_id == "Section 37"

    def test_unknown_jurisdiction_returns_parse_error(
        self, tiny_gold: GoldSet, tmp_path: Path, fake_embedder: None
    ) -> None:
        out = tmp_path / "tiny"
        build_index(tiny_gold, EMBEDDER_NAME, out)
        client = RetrievalClient(index_path=out, embedder_name=EMBEDDER_NAME)
        messages = PromptMessages(
            system="s", user="u",
            source_clause_text="Security of processing.",
            target_jurisdiction="my",  # not in tiny_gold
        )
        resp = client.generate(messages, run_id="r", batch_id="b")
        assert resp.top1 is None
        assert resp.parse_error is not None
        assert "no indexed entries" in resp.parse_error

    def test_missing_source_clause_returns_parse_error(
        self, tiny_gold: GoldSet, tmp_path: Path, fake_embedder: None
    ) -> None:
        out = tmp_path / "tiny"
        build_index(tiny_gold, EMBEDDER_NAME, out)
        client = RetrievalClient(index_path=out, embedder_name=EMBEDDER_NAME)
        # API-client-shaped PromptMessages (no source_clause_text / target_jurisdiction)
        messages = PromptMessages(system="s", user="u")
        resp = client.generate(messages, run_id="r", batch_id="b")
        assert resp.top1 is None
        assert resp.parse_error is not None
        assert "source_clause_text" in resp.parse_error

    def test_score_threshold_below_returns_top1_none(
        self, tiny_gold: GoldSet, tmp_path: Path, fake_embedder: None
    ) -> None:
        out = tmp_path / "tiny"
        build_index(tiny_gold, EMBEDDER_NAME, out)
        # Threshold above 1.0 forces every query to fall below.
        client = RetrievalClient(
            index_path=out, embedder_name=EMBEDDER_NAME, score_threshold=2.0
        )
        messages = PromptMessages(
            system="s", user="u",
            source_clause_text="Security of processing.",
            target_jurisdiction="sg",
        )
        resp = client.generate(messages, run_id="r", batch_id="b")
        assert resp.top1 is None
        assert resp.parse_error is not None
        assert "no confident retrieval match" in resp.parse_error
        assert "cosine=" in resp.parse_error

    def test_provider_is_retrieval(
        self, tiny_gold: GoldSet, tmp_path: Path, fake_embedder: None
    ) -> None:
        out = tmp_path / "tiny"
        build_index(tiny_gold, EMBEDDER_NAME, out)
        client = RetrievalClient(index_path=out, embedder_name=EMBEDDER_NAME)
        assert client.provider == "retrieval"
        assert client.model == f"retrieval/{EMBEDDER_NAME}"
