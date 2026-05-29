"""Bidirectional cross-check guards.

Pins each of the seven `BidirectionalStatus` outcomes with the smallest
fixture that triggers it, plus the citation_id normalisation roundtrip.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from daccord.ensemble.bidirectional import (
    build_framework_lookup,
    compute_bidirectional_for_pair,
    load_bidirectional_for_pair,
    reverse_pair_name,
)
from daccord.ensemble.schema import EnsembleCandidate, Tier, TieredPair


def _raw(
    source_id: str,
    source_framework: str,
    source_citation_id: str,
    target_framework: str = "pdpa_sg",
    citation_id: str = "13",
) -> EnsembleCandidate:
    return EnsembleCandidate(
        source_id=source_id,
        source_jurisdiction="x",
        source_framework=source_framework,
        source_citation_id=source_citation_id,
        source_mechanism="...",
        target_jurisdiction="y",
        target_framework=target_framework,
        model="m1",
        citation_id=citation_id,
        target_mechanism="...",
        mapping_justification="...",
    )


def _tp(
    source_id: str,
    source_framework: str,
    source_citation_id: str,
    target_framework: str,
    tier: Tier,
    consensus: str,
    *,
    agreement_score: float | None = None,
) -> TieredPair:
    if agreement_score is None:
        agreement_score = (
            1.0 if tier == "HIGH" else 0.75 if tier == "MED" else 0.25 if tier == "LOW" else -1.0
        )
    valid_count = 0 if tier == "SALVAGE" else 4
    if tier == "SALVAGE":
        consensus_vote = 0
    elif tier == "HIGH":
        consensus_vote = 4
    elif tier == "MED":
        consensus_vote = 3
    else:
        consensus_vote = 1
    return TieredPair(
        source_id=source_id,
        source_jurisdiction="x",
        source_framework=source_framework,
        source_citation_id=source_citation_id,
        source_mechanism="...",
        target_jurisdiction="y",
        target_framework=target_framework,
        tier=tier,
        consensus_citation_id=consensus,
        valid_vote_count=valid_count,
        consensus_vote_count=consensus_vote,
        agreement_score=agreement_score,
        votes=[],
    )


def _write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.write_text("\n".join(r.model_dump_json() for r in rows) + "\n", encoding="utf-8")


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Default raw fixture: two source clauses in each direction.

    - gdpr-1 / source_citation_id="32"
    - gdpr-2 / source_citation_id="5"
    - pdpa_sg-1 / source_citation_id="13"
    - pdpa_sg-2 / source_citation_id="11"
    """
    raw = tmp_path / "raw"
    tiered = tmp_path / "tiered"
    _write_jsonl(
        raw / "gdpr__pdpa_sg__m1.jsonl",
        [
            _raw("gdpr-1", "gdpr", "32"),
            _raw("gdpr-2", "gdpr", "5"),
        ],
    )
    _write_jsonl(
        raw / "pdpa_sg__gdpr__m1.jsonl",
        [
            _raw("pdpa_sg-1", "pdpa_sg", "13", target_framework="gdpr", citation_id="32"),
            _raw("pdpa_sg-2", "pdpa_sg", "11", target_framework="gdpr", citation_id="5"),
        ],
    )
    return raw, tiered


def test_reverse_pair_name() -> None:
    assert reverse_pair_name("gdpr__pdpa_sg") == "pdpa_sg__gdpr"
    assert reverse_pair_name("uk_gdpr__dpa_2012_ph") == "dpa_2012_ph__uk_gdpr"


class TestBuildFrameworkLookup:
    def test_returns_normalized_citation_to_source_id(self, dirs: tuple[Path, Path]) -> None:
        raw, _ = dirs
        lookup = build_framework_lookup("gdpr", raw)
        assert lookup["32"] == "gdpr-1"
        assert lookup["5"] == "gdpr-2"

    def test_missing_framework_returns_empty(self, dirs: tuple[Path, Path]) -> None:
        raw, _ = dirs
        assert build_framework_lookup("nope", raw) == {}


class TestBidirectionalStatuses:
    def test_consistent_round_trip(self, dirs: tuple[Path, Path]) -> None:
        raw, tiered = dirs
        _write_jsonl(
            tiered / "gdpr__pdpa_sg.jsonl",
            [_tp("gdpr-1", "gdpr", "32", "pdpa_sg", "HIGH", "13")],
        )
        _write_jsonl(
            tiered / "pdpa_sg__gdpr.jsonl",
            [_tp("pdpa_sg-1", "pdpa_sg", "13", "gdpr", "HIGH", "32")],
        )

        results = compute_bidirectional_for_pair("gdpr__pdpa_sg", tiered, raw)
        assert len(results) == 1
        r = results[0]
        assert r.status == "consistent"
        assert r.source_id == "gdpr-1"
        assert r.reverse_pair == "pdpa_sg__gdpr"
        assert r.reverse_source_id == "pdpa_sg-1"
        assert r.reverse_consensus == "32"
        assert r.reverse_tier == "HIGH"
        assert r.reverse_agreement_score == 1.0

    def test_inconsistent_round_trip(self, dirs: tuple[Path, Path]) -> None:
        raw, tiered = dirs
        # Forward gdpr-1 says target = "13" (pdpa_sg-1). But reverse pdpa_sg-1
        # says target = "5" (gdpr-2). Mismatch — inconsistent.
        _write_jsonl(
            tiered / "gdpr__pdpa_sg.jsonl",
            [_tp("gdpr-1", "gdpr", "32", "pdpa_sg", "HIGH", "13")],
        )
        _write_jsonl(
            tiered / "pdpa_sg__gdpr.jsonl",
            [_tp("pdpa_sg-1", "pdpa_sg", "13", "gdpr", "HIGH", "5")],
        )

        results = compute_bidirectional_for_pair("gdpr__pdpa_sg", tiered, raw)
        assert results[0].status == "inconsistent"
        assert results[0].reverse_consensus == "5"

    def test_reverse_LOW_is_unknown(self, dirs: tuple[Path, Path]) -> None:
        raw, tiered = dirs
        _write_jsonl(
            tiered / "gdpr__pdpa_sg.jsonl",
            [_tp("gdpr-1", "gdpr", "32", "pdpa_sg", "HIGH", "13")],
        )
        # Reverse is LOW — no usable signal.
        _write_jsonl(
            tiered / "pdpa_sg__gdpr.jsonl",
            [_tp("pdpa_sg-1", "pdpa_sg", "13", "gdpr", "LOW", "32")],
        )

        results = compute_bidirectional_for_pair("gdpr__pdpa_sg", tiered, raw)
        assert results[0].status == "reverse_unknown"
        # Reverse data is still surfaced so the labeler can show it as a soft hint.
        assert results[0].reverse_consensus == "32"
        assert results[0].reverse_tier == "LOW"

    def test_reverse_SALVAGE_is_unknown(self, dirs: tuple[Path, Path]) -> None:
        raw, tiered = dirs
        _write_jsonl(
            tiered / "gdpr__pdpa_sg.jsonl",
            [_tp("gdpr-1", "gdpr", "32", "pdpa_sg", "HIGH", "13")],
        )
        _write_jsonl(
            tiered / "pdpa_sg__gdpr.jsonl",
            [_tp("pdpa_sg-1", "pdpa_sg", "13", "gdpr", "SALVAGE", "")],
        )

        results = compute_bidirectional_for_pair("gdpr__pdpa_sg", tiered, raw)
        assert results[0].status == "reverse_unknown"
        assert results[0].reverse_tier == "SALVAGE"

    def test_missing_in_registry(self, dirs: tuple[Path, Path]) -> None:
        raw, tiered = dirs
        # Forward consensus "999" doesn't appear in pdpa_sg's source clauses.
        _write_jsonl(
            tiered / "gdpr__pdpa_sg.jsonl",
            [_tp("gdpr-1", "gdpr", "32", "pdpa_sg", "HIGH", "999")],
        )
        _write_jsonl(
            tiered / "pdpa_sg__gdpr.jsonl",
            [_tp("pdpa_sg-1", "pdpa_sg", "13", "gdpr", "HIGH", "32")],
        )

        results = compute_bidirectional_for_pair("gdpr__pdpa_sg", tiered, raw)
        assert results[0].status == "missing_in_registry"
        assert results[0].reverse_source_id is None

    def test_missing_reverse_row(self, dirs: tuple[Path, Path]) -> None:
        raw, tiered = dirs
        # Forward maps to pdpa_sg-1 (citation "13"), but reverse tiered has
        # only pdpa_sg-2 — pdpa_sg-1 wasn't tiered for some reason.
        _write_jsonl(
            tiered / "gdpr__pdpa_sg.jsonl",
            [_tp("gdpr-1", "gdpr", "32", "pdpa_sg", "HIGH", "13")],
        )
        _write_jsonl(
            tiered / "pdpa_sg__gdpr.jsonl",
            [_tp("pdpa_sg-2", "pdpa_sg", "11", "gdpr", "HIGH", "5")],
        )

        results = compute_bidirectional_for_pair("gdpr__pdpa_sg", tiered, raw)
        assert results[0].status == "missing_reverse_row"
        assert results[0].reverse_source_id == "pdpa_sg-1"

    def test_missing_reverse_pair(self, dirs: tuple[Path, Path]) -> None:
        raw, tiered = dirs
        _write_jsonl(
            tiered / "gdpr__pdpa_sg.jsonl",
            [_tp("gdpr-1", "gdpr", "32", "pdpa_sg", "HIGH", "13")],
        )
        # No reverse pair file at all.

        results = compute_bidirectional_for_pair("gdpr__pdpa_sg", tiered, raw)
        assert results[0].status == "missing_reverse_pair"
        assert results[0].reverse_pair == "pdpa_sg__gdpr"

    def test_no_forward_consensus(self, dirs: tuple[Path, Path]) -> None:
        raw, tiered = dirs
        # Forward is SALVAGE — empty consensus, nothing to look up.
        _write_jsonl(
            tiered / "gdpr__pdpa_sg.jsonl",
            [_tp("gdpr-1", "gdpr", "32", "pdpa_sg", "SALVAGE", "")],
        )
        _write_jsonl(tiered / "pdpa_sg__gdpr.jsonl", [])

        results = compute_bidirectional_for_pair("gdpr__pdpa_sg", tiered, raw)
        assert results[0].status == "no_forward_consensus"
        assert results[0].reverse_source_id is None


class TestNormalisation:
    def test_consistent_with_section_prefix_normalises_equal(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        tiered = tmp_path / "tiered"
        # Raw forward: source_citation_id = "Section 32" (will normalise to "32")
        # Raw reverse: source_citation_id = "Sec 13" (will normalise to "13")
        _write_jsonl(
            raw / "gdpr__pdpa_sg__m1.jsonl",
            [_raw("gdpr-1", "gdpr", "Section 32")],
        )
        _write_jsonl(
            raw / "pdpa_sg__gdpr__m1.jsonl",
            [_raw("pdpa_sg-1", "pdpa_sg", "Sec 13", target_framework="gdpr", citation_id="32")],
        )
        # Tiered forward says consensus "13" (matches normalised "Sec 13").
        # Tiered reverse says consensus "32" (matches normalised "Section 32").
        _write_jsonl(
            tiered / "gdpr__pdpa_sg.jsonl",
            [_tp("gdpr-1", "gdpr", "Section 32", "pdpa_sg", "HIGH", "13")],
        )
        _write_jsonl(
            tiered / "pdpa_sg__gdpr.jsonl",
            [_tp("pdpa_sg-1", "pdpa_sg", "Sec 13", "gdpr", "HIGH", "32")],
        )

        results = compute_bidirectional_for_pair("gdpr__pdpa_sg", tiered, raw)
        assert results[0].status == "consistent"


class TestOverlayLoader:
    def test_load_bidirectional_for_pair(self, dirs: tuple[Path, Path]) -> None:
        raw, tiered = dirs
        _write_jsonl(
            tiered / "gdpr__pdpa_sg.jsonl",
            [_tp("gdpr-1", "gdpr", "32", "pdpa_sg", "HIGH", "13")],
        )
        _write_jsonl(
            tiered / "pdpa_sg__gdpr.jsonl",
            [_tp("pdpa_sg-1", "pdpa_sg", "13", "gdpr", "HIGH", "32")],
        )

        results = compute_bidirectional_for_pair("gdpr__pdpa_sg", tiered, raw)
        # Persist results
        out_dir = tiered.parent / "bidirectional"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "gdpr__pdpa_sg.jsonl"
        out_path.write_text(
            "\n".join(r.model_dump_json() for r in results) + "\n", encoding="utf-8"
        )

        loaded = load_bidirectional_for_pair(out_dir, "gdpr__pdpa_sg")
        assert "gdpr-1" in loaded
        assert loaded["gdpr-1"].status == "consistent"

    def test_load_bidirectional_returns_empty_when_dir_absent(self, tmp_path: Path) -> None:
        assert load_bidirectional_for_pair(None, "gdpr__pdpa_sg") == {}
        assert load_bidirectional_for_pair(tmp_path / "nope", "gdpr__pdpa_sg") == {}
