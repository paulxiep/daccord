"""Tier-7C labeler queue logic guards.

The Streamlit UI is thin glue around `build_review_queue`. These tests pin
the filtering + sorting semantics (tier filter, bidirectional filter, sort
order, resume-by-source_id) on minimal fixtures.
"""

from __future__ import annotations

from pathlib import Path

from labeler.queue import (
    build_review_queue,
    list_pair_files,
    progress_summary,
    vote_choices,
)

from daccord.ensemble.bidirectional import BidirectionalResult
from daccord.ensemble.schema import ModelVote, Tier, TieredPair
from daccord.ensemble.validated import ValidatedPair, append_validation


def _vote(model: str, raw: str, normalised: str | None = None) -> ModelVote:
    return ModelVote(
        model=model,
        citation_id_raw=raw,
        citation_id_normalized=normalised if normalised is not None else raw,
        target_mechanism="...",
        mapping_justification="...",
    )


def _tp(
    source_id: str,
    tier: Tier,
    *,
    agreement_score: float = 1.0,
    valid_vote_count: int = 4,
    consensus_vote_count: int = 4,
    consensus: str = "24",
    votes: list[ModelVote] | None = None,
) -> TieredPair:
    return TieredPair(
        source_id=source_id,
        source_jurisdiction="eu",
        source_framework="gdpr",
        source_citation_id="Art. 32",
        source_mechanism="...",
        target_jurisdiction="sg",
        target_framework="pdpa_sg",
        tier=tier,
        consensus_citation_id=consensus,
        valid_vote_count=valid_vote_count,
        consensus_vote_count=consensus_vote_count,
        agreement_score=agreement_score,
        votes=votes or [_vote("m1", "24"), _vote("m2", "24")],
    )


def _bi(
    source_id: str,
    status: str,
    *,
    reverse_consensus: str | None = "32",
    reverse_tier: Tier | None = "HIGH",
    reverse_agreement: float | None = 1.0,
) -> BidirectionalResult:
    return BidirectionalResult(
        source_id=source_id,
        forward_consensus="24",
        reverse_pair="pdpa_sg__gdpr",
        reverse_source_id=f"reverse-{source_id}",
        reverse_consensus=reverse_consensus,
        reverse_tier=reverse_tier,
        reverse_agreement_score=reverse_agreement,
        status=status,  # type: ignore[arg-type]
    )


def _write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.write_text("\n".join(r.model_dump_json() for r in rows) + "\n", encoding="utf-8")


def _make_validated(source_id: str) -> ValidatedPair:
    return ValidatedPair(
        source_id=source_id,
        source_jurisdiction="eu",
        source_framework="gdpr",
        target_jurisdiction="sg",
        target_framework="pdpa_sg",
        chosen_citation_id="24",
        human_note="",
        reviewer="tester",
        reviewed_at="2026-05-27T12:00:00Z",
        tier_at_review="MED",
        agreement_score_at_review=0.75,
    )


class TestTierFilter:
    def test_default_med_low_salvage_skips_high(self, tmp_path: Path) -> None:
        tiered = tmp_path / "tiered" / "gdpr__pdpa_sg.jsonl"
        validated = tmp_path / "validated" / "gdpr__pdpa_sg.jsonl"
        _write_jsonl(
            tiered,
            [
                _tp("h-1", "HIGH"),
                _tp("m-1", "MED", agreement_score=0.75, consensus_vote_count=3),
                _tp("l-1", "LOW", agreement_score=0.25, consensus_vote_count=1),
                _tp(
                    "s-1",
                    "SALVAGE",
                    agreement_score=-1.0,
                    valid_vote_count=0,
                    consensus_vote_count=0,
                    consensus="",
                ),
            ],
        )

        queue = build_review_queue(
            tiered_path=tiered,
            validated_path=validated,
            tier_filter=["MED", "LOW", "SALVAGE"],
        )
        assert {q.pair.source_id for q in queue} == {"m-1", "l-1", "s-1"}


class TestResumeByReviewedId:
    def test_already_reviewed_pairs_are_filtered(self, tmp_path: Path) -> None:
        tiered = tmp_path / "tiered" / "gdpr__pdpa_sg.jsonl"
        validated = tmp_path / "validated" / "gdpr__pdpa_sg.jsonl"
        _write_jsonl(
            tiered,
            [
                _tp("m-1", "MED", agreement_score=0.75, consensus_vote_count=3),
                _tp("m-2", "MED", agreement_score=0.75, consensus_vote_count=3),
                _tp("m-3", "MED", agreement_score=0.75, consensus_vote_count=3),
            ],
        )
        append_validation(validated, _make_validated("m-1"))
        append_validation(validated, _make_validated("m-3"))

        queue = build_review_queue(
            tiered_path=tiered,
            validated_path=validated,
            tier_filter=["MED"],
        )
        assert [q.pair.source_id for q in queue] == ["m-2"]


class TestBidirectionalFilter:
    def test_filter_to_inconsistent_only(self, tmp_path: Path) -> None:
        tiered = tmp_path / "tiered" / "gdpr__pdpa_sg.jsonl"
        validated = tmp_path / "validated" / "gdpr__pdpa_sg.jsonl"
        bidirectional = tmp_path / "bidirectional"
        _write_jsonl(
            tiered,
            [
                _tp("m-1", "MED", agreement_score=0.75, consensus_vote_count=3),
                _tp("m-2", "MED", agreement_score=0.75, consensus_vote_count=3),
                _tp("m-3", "MED", agreement_score=0.75, consensus_vote_count=3),
            ],
        )
        _write_jsonl(
            bidirectional / "gdpr__pdpa_sg.jsonl",
            [
                _bi("m-1", "consistent"),
                _bi("m-2", "inconsistent"),
                _bi("m-3", "reverse_unknown"),
            ],
        )

        queue = build_review_queue(
            tiered_path=tiered,
            validated_path=validated,
            bidirectional_dir=bidirectional,
            tier_filter=["MED"],
            bidirectional_filter=["inconsistent"],
        )
        assert [q.pair.source_id for q in queue] == ["m-2"]
        assert queue[0].bidirectional is not None
        assert queue[0].bidirectional.status == "inconsistent"

    def test_default_no_filter_returns_all_with_overlay(self, tmp_path: Path) -> None:
        tiered = tmp_path / "tiered" / "gdpr__pdpa_sg.jsonl"
        validated = tmp_path / "validated" / "gdpr__pdpa_sg.jsonl"
        bidirectional = tmp_path / "bidirectional"
        _write_jsonl(
            tiered,
            [
                _tp("m-1", "MED", agreement_score=0.75, consensus_vote_count=3),
                _tp("m-2", "MED", agreement_score=0.75, consensus_vote_count=3),
            ],
        )
        _write_jsonl(
            bidirectional / "gdpr__pdpa_sg.jsonl",
            [_bi("m-1", "consistent")],
        )

        queue = build_review_queue(
            tiered_path=tiered,
            validated_path=validated,
            bidirectional_dir=bidirectional,
            tier_filter=["MED"],
            bidirectional_filter=None,
        )
        assert {q.pair.source_id for q in queue} == {"m-1", "m-2"}
        # m-1 has overlay; m-2 doesn't.
        bi_by_sid = {q.pair.source_id: q.bidirectional for q in queue}
        assert bi_by_sid["m-1"] is not None
        assert bi_by_sid["m-2"] is None


class TestSortOrder:
    def test_confidence_desc_orders_high_score_first(self, tmp_path: Path) -> None:
        tiered = tmp_path / "tiered" / "gdpr__pdpa_sg.jsonl"
        validated = tmp_path / "validated" / "gdpr__pdpa_sg.jsonl"
        _write_jsonl(
            tiered,
            [
                _tp("a", "MED", agreement_score=0.6, consensus_vote_count=2),
                _tp("b", "MED", agreement_score=0.9, consensus_vote_count=3),
                _tp("c", "MED", agreement_score=0.75, consensus_vote_count=3),
            ],
        )

        queue = build_review_queue(
            tiered_path=tiered,
            validated_path=validated,
            tier_filter=["MED"],
            sort_order="confidence-desc",
        )
        assert [q.pair.source_id for q in queue] == ["b", "c", "a"]

    def test_source_id_sort_is_deterministic(self, tmp_path: Path) -> None:
        tiered = tmp_path / "tiered" / "gdpr__pdpa_sg.jsonl"
        validated = tmp_path / "validated" / "gdpr__pdpa_sg.jsonl"
        _write_jsonl(
            tiered,
            [
                _tp("c", "MED", agreement_score=0.9, consensus_vote_count=3),
                _tp("a", "MED", agreement_score=0.6, consensus_vote_count=2),
                _tp("b", "MED", agreement_score=0.75, consensus_vote_count=3),
            ],
        )

        queue = build_review_queue(
            tiered_path=tiered,
            validated_path=validated,
            tier_filter=["MED"],
            sort_order="source-id",
        )
        assert [q.pair.source_id for q in queue] == ["a", "b", "c"]


class TestListPairFiles:
    def test_reports_reviewed_and_bidirectional_counts(self, tmp_path: Path) -> None:
        tiered_dir = tmp_path / "tiered"
        validated_dir = tmp_path / "validated"
        bidirectional_dir = tmp_path / "bidirectional"
        _write_jsonl(
            tiered_dir / "gdpr__pdpa_sg.jsonl",
            [
                _tp("m-1", "MED", agreement_score=0.75, consensus_vote_count=3),
                _tp("m-2", "MED", agreement_score=0.75, consensus_vote_count=3),
            ],
        )
        append_validation(validated_dir / "gdpr__pdpa_sg.jsonl", _make_validated("m-1"))
        _write_jsonl(
            bidirectional_dir / "gdpr__pdpa_sg.jsonl",
            [_bi("m-1", "consistent"), _bi("m-2", "inconsistent")],
        )

        files = list_pair_files(tiered_dir, validated_dir, bidirectional_dir)
        assert len(files) == 1
        f = files[0]
        assert f.name == "gdpr__pdpa_sg"
        assert f.total == 2
        assert f.reviewed == 1
        assert f.bidirectional_consistent == 1
        assert f.bidirectional_inconsistent == 1


class TestProgressSummary:
    def test_aggregates_across_pairs(self, tmp_path: Path) -> None:
        # Two pair files, three reviewed total out of five.
        tiered_dir = tmp_path / "tiered"
        validated_dir = tmp_path / "validated"
        _write_jsonl(
            tiered_dir / "gdpr__pdpa_sg.jsonl",
            [_tp(f"m-{i}", "MED", agreement_score=0.75, consensus_vote_count=3) for i in range(3)],
        )
        _write_jsonl(
            tiered_dir / "bdsg__loi_il.jsonl",
            [_tp(f"n-{i}", "MED", agreement_score=0.75, consensus_vote_count=3) for i in range(2)],
        )
        append_validation(validated_dir / "gdpr__pdpa_sg.jsonl", _make_validated("m-0"))
        append_validation(validated_dir / "gdpr__pdpa_sg.jsonl", _make_validated("m-1"))
        append_validation(validated_dir / "bdsg__loi_il.jsonl", _make_validated("n-0"))

        files = list_pair_files(tiered_dir, validated_dir, None)
        progress = progress_summary(files)
        assert progress == {"total": 5, "reviewed": 3, "remaining": 2}


class TestVoteChoices:
    def test_dedupes_and_orders_by_vote_count(self) -> None:
        row = _tp(
            "m-1",
            "MED",
            votes=[
                _vote("m1", "24"),
                _vote("m2", "24"),
                _vote("m3", "26"),
                _vote("m4", ""),
            ],
        )
        choices = vote_choices(row)
        # Two real options (24 with 2 votes, 26 with 1) + the no-analog sentinel.
        values = [v for v, _ in choices]
        assert "24" in values
        assert "26" in values
        assert "" in values  # no-analog sentinel
        # "24" should come first (more votes).
        assert values[0] == "24"
