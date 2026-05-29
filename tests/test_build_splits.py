"""Tier-7B splits guards.

Pins the jurisdiction-disjoint invariant, the overlay-driven MED inclusion
rule, the HIGH-rejection-via-overlay path, and SHA256 stability for
MLflow reproducibility.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from daccord.ensemble.schema import ModelVote, Tier, TieredPair
from daccord.ensemble.splits import (
    DEFAULT_TEST_JURISDICTIONS,
    DEFAULT_VAL_JURISDICTIONS,
    build_splits,
)


def _make_pair(
    source_id: str,
    source_jurisdiction: str,
    source_framework: str,
    tier: Tier,
    *,
    target_framework: str = "pdpa_sg",
    target_jurisdiction: str = "sg",
    agreement_score: float = 1.0,
    valid_vote_count: int = 4,
    consensus_vote_count: int = 4,
    consensus_citation_id: str = "24",
    rag_concurs: bool = False,
    rag_vote_citation_id: str = "",
) -> TieredPair:
    return TieredPair(
        source_id=source_id,
        source_jurisdiction=source_jurisdiction,
        source_framework=source_framework,
        source_citation_id="Art. 32",
        source_mechanism="...",
        target_jurisdiction=target_jurisdiction,
        target_framework=target_framework,
        tier=tier,
        consensus_citation_id=consensus_citation_id,
        valid_vote_count=valid_vote_count,
        consensus_vote_count=consensus_vote_count,
        agreement_score=agreement_score,
        votes=[
            ModelVote(
                model="m1",
                citation_id_raw="24",
                citation_id_normalized="24",
                target_mechanism="...",
                mapping_justification="...",
            )
        ],
        rag_concurs=rag_concurs,
        rag_vote_citation_id=rag_vote_citation_id,
    )


def _write_tiered(tiered_dir: Path, pair_name: str, rows: list[TieredPair]) -> None:
    tiered_dir.mkdir(parents=True, exist_ok=True)
    path = tiered_dir / f"{pair_name}.jsonl"
    path.write_text("\n".join(r.model_dump_json() for r in rows) + "\n", encoding="utf-8")


def _write_overlay(validated_dir: Path, pair_name: str, rows: list[dict]) -> None:
    validated_dir.mkdir(parents=True, exist_ok=True)
    path = validated_dir / f"{pair_name}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


class TestJurisdictionDisjoint:
    def test_default_partition_keeps_jurisdictions_disjoint(self, tmp_path: Path) -> None:
        tiered = tmp_path / "tiered"
        # One HIGH row per source_jurisdiction
        for juris, framework in [
            ("de", "bdsg"),
            ("eu", "gdpr"),
            ("uk", "uk_gdpr"),
            ("fr", "loi_il"),
            ("sg", "pdpa_sg"),
            ("my", "pdpa_my"),
            ("ph", "dpa_2012_ph"),
            ("th", "pdpa_th"),
        ]:
            _write_tiered(
                tiered,
                f"{framework}__pdpa_sg",
                [_make_pair(f"{framework}-1", juris, framework, "HIGH")],
            )

        m = build_splits(tiered_dir=tiered, out_dir=tmp_path / "splits")

        all_jurisdictions = (
            set(m.train.source_jurisdictions)
            | set(m.val.source_jurisdictions)
            | set(m.test.source_jurisdictions)
        )
        # No jurisdiction in more than one split.
        assert (
            set(m.train.source_jurisdictions).isdisjoint(m.val.source_jurisdictions)
            and set(m.train.source_jurisdictions).isdisjoint(m.test.source_jurisdictions)
            and set(m.val.source_jurisdictions).isdisjoint(m.test.source_jurisdictions)
        )
        # Defaults: test = {th, ph}, val = {my}, train = remainder
        assert set(m.test.source_jurisdictions) == set(DEFAULT_TEST_JURISDICTIONS)
        assert set(m.val.source_jurisdictions) == set(DEFAULT_VAL_JURISDICTIONS)
        assert "de" in all_jurisdictions  # sanity

    def test_overlapping_val_and_test_raises(self, tmp_path: Path) -> None:
        tiered = tmp_path / "tiered"
        _write_tiered(tiered, "gdpr__pdpa_sg", [_make_pair("g-1", "eu", "gdpr", "HIGH")])
        with pytest.raises(ValueError, match="overlap"):
            build_splits(
                tiered_dir=tiered,
                out_dir=tmp_path / "splits",
                val_jurisdictions=["th"],
                test_jurisdictions=["th"],
            )


class TestTierFloor:
    def test_high_only_default_excludes_unvalidated_med(self, tmp_path: Path) -> None:
        tiered = tmp_path / "tiered"
        rows = [
            _make_pair("high-1", "de", "bdsg", "HIGH"),
            _make_pair("med-1", "de", "bdsg", "MED", agreement_score=0.75),
            _make_pair("low-1", "de", "bdsg", "LOW", agreement_score=0.5),
        ]
        _write_tiered(tiered, "bdsg__gdpr", rows)

        m = build_splits(tiered_dir=tiered, out_dir=tmp_path / "splits")
        # de → train; only the HIGH row makes it.
        assert m.train.count == 1
        assert m.val.count == 0
        assert m.test.count == 0

    def test_med_floor_includes_validated_med(self, tmp_path: Path) -> None:
        tiered = tmp_path / "tiered"
        validated = tmp_path / "validated"
        rows = [
            _make_pair("high-1", "de", "bdsg", "HIGH"),
            _make_pair("med-1", "de", "bdsg", "MED", agreement_score=0.75),
            _make_pair("med-2", "de", "bdsg", "MED", agreement_score=0.75),
            _make_pair("low-1", "de", "bdsg", "LOW", agreement_score=0.5),
        ]
        _write_tiered(tiered, "bdsg__gdpr", rows)
        # Only med-1 is validated; med-2 is not, low-1 is not.
        _write_overlay(
            validated,
            "bdsg__gdpr",
            [{"source_id": "med-1", "chosen_citation_id": "24"}],
        )

        m = build_splits(
            tiered_dir=tiered,
            out_dir=tmp_path / "splits",
            validated_dir=validated,
            tier_floor="MED",
        )
        # HIGH (high-1) + validated MED (med-1) → 2 rows. med-2 unvalidated; low-1 below floor.
        assert m.train.count == 2

    def test_med_floor_skips_low_even_when_validated(self, tmp_path: Path) -> None:
        tiered = tmp_path / "tiered"
        validated = tmp_path / "validated"
        rows = [_make_pair("low-1", "de", "bdsg", "LOW", agreement_score=0.5)]
        _write_tiered(tiered, "bdsg__gdpr", rows)
        # LOW is validated, but floor is MED → still excluded.
        _write_overlay(
            validated,
            "bdsg__gdpr",
            [{"source_id": "low-1", "chosen_citation_id": "24"}],
        )

        m = build_splits(
            tiered_dir=tiered,
            out_dir=tmp_path / "splits",
            validated_dir=validated,
            tier_floor="MED",
        )
        assert m.train.count == 0

    def test_no_analog_overlay_excludes_high(self, tmp_path: Path) -> None:
        # Reviewer can override a HIGH row by explicitly marking no-analog
        # (chosen_citation_id == ""). The HIGH then drops from gold.
        tiered = tmp_path / "tiered"
        validated = tmp_path / "validated"
        rows = [_make_pair("high-1", "de", "bdsg", "HIGH")]
        _write_tiered(tiered, "bdsg__gdpr", rows)
        _write_overlay(
            validated,
            "bdsg__gdpr",
            [{"source_id": "high-1", "chosen_citation_id": ""}],
        )

        m = build_splits(
            tiered_dir=tiered,
            out_dir=tmp_path / "splits",
            validated_dir=validated,
        )
        assert m.train.count == 0


class TestSha256Stability:
    def test_manifest_sha_is_stable_across_reruns(self, tmp_path: Path) -> None:
        tiered = tmp_path / "tiered"
        rows = [
            _make_pair("eu-1", "eu", "gdpr", "HIGH"),
            _make_pair("de-1", "de", "bdsg", "HIGH"),
        ]
        _write_tiered(tiered, "gdpr__bdsg", rows)

        m1 = build_splits(tiered_dir=tiered, out_dir=tmp_path / "splits1")
        m2 = build_splits(tiered_dir=tiered, out_dir=tmp_path / "splits2")
        assert m1.tiered_input_sha256 == m2.tiered_input_sha256
        assert m1.validated_input_sha256 is None
        assert m2.validated_input_sha256 is None

    def test_validated_input_sha_recorded_when_overlay_present(self, tmp_path: Path) -> None:
        tiered = tmp_path / "tiered"
        validated = tmp_path / "validated"
        _write_tiered(tiered, "gdpr__bdsg", [_make_pair("eu-1", "eu", "gdpr", "HIGH")])
        _write_overlay(
            validated,
            "gdpr__bdsg",
            [{"source_id": "fake-1", "chosen_citation_id": "x"}],
        )

        m = build_splits(
            tiered_dir=tiered,
            out_dir=tmp_path / "splits",
            validated_dir=validated,
        )
        assert m.validated_input_sha256 is not None
        assert len(m.validated_input_sha256) == 64  # sha256 hex


class TestDryRun:
    def test_dry_run_does_not_write_files(self, tmp_path: Path) -> None:
        tiered = tmp_path / "tiered"
        out = tmp_path / "splits"
        _write_tiered(tiered, "gdpr__bdsg", [_make_pair("eu-1", "eu", "gdpr", "HIGH")])

        m = build_splits(tiered_dir=tiered, out_dir=out, write=False)
        assert m.train.count == 1
        assert not out.exists()


class TestNoInputs:
    def test_missing_tiered_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            build_splits(tiered_dir=tmp_path / "nope", out_dir=tmp_path / "splits")


def _write_bidirectional(bidirectional_dir: Path, pair_name: str, rows: list[dict]) -> None:
    bidirectional_dir.mkdir(parents=True, exist_ok=True)
    path = bidirectional_dir / f"{pair_name}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


class TestBidirectionalPromotion:
    def test_promote_flag_auto_includes_consistent_med(self, tmp_path: Path) -> None:
        tiered = tmp_path / "tiered"
        bidirectional = tmp_path / "bidirectional"
        rows = [
            _make_pair("high-1", "de", "bdsg", "HIGH"),
            _make_pair("med-1", "de", "bdsg", "MED", agreement_score=0.75),
            _make_pair("med-2", "de", "bdsg", "MED", agreement_score=0.75),
        ]
        _write_tiered(tiered, "bdsg__pdpa_sg", rows)
        # med-1 is bidirectionally consistent; med-2 is not.
        _write_bidirectional(
            bidirectional,
            "bdsg__pdpa_sg",
            [
                {"source_id": "high-1", "status": "consistent"},
                {"source_id": "med-1", "status": "consistent"},
                {"source_id": "med-2", "status": "inconsistent"},
            ],
        )

        m = build_splits(
            tiered_dir=tiered,
            out_dir=tmp_path / "splits",
            bidirectional_dir=bidirectional,
            promote_bidirectional_consistent=True,
        )
        # HIGH (high-1) + auto-promoted MED (med-1) = 2 rows in train.
        assert m.train.count == 2
        assert m.bidirectional_input_sha256 is not None
        assert m.promote_bidirectional_consistent is True

    def test_promote_flag_off_means_no_med_auto_promotion(self, tmp_path: Path) -> None:
        tiered = tmp_path / "tiered"
        bidirectional = tmp_path / "bidirectional"
        _write_tiered(
            tiered,
            "bdsg__pdpa_sg",
            [_make_pair("med-1", "de", "bdsg", "MED", agreement_score=0.75)],
        )
        _write_bidirectional(
            bidirectional,
            "bdsg__pdpa_sg",
            [{"source_id": "med-1", "status": "consistent"}],
        )

        m = build_splits(
            tiered_dir=tiered,
            out_dir=tmp_path / "splits",
            bidirectional_dir=bidirectional,
            promote_bidirectional_consistent=False,
        )
        # MED not promoted (no validated overlay, promote flag off) → 0 rows.
        assert m.train.count == 0
        # But the manifest still records the bidirectional input SHA so
        # the operator can audit which overlay was on disk.
        assert m.bidirectional_input_sha256 is not None
        assert m.promote_bidirectional_consistent is False

    def test_low_is_not_auto_promoted_even_when_consistent(self, tmp_path: Path) -> None:
        tiered = tmp_path / "tiered"
        bidirectional = tmp_path / "bidirectional"
        _write_tiered(
            tiered,
            "bdsg__pdpa_sg",
            [_make_pair("low-1", "de", "bdsg", "LOW", agreement_score=0.5)],
        )
        _write_bidirectional(
            bidirectional,
            "bdsg__pdpa_sg",
            [{"source_id": "low-1", "status": "consistent"}],
        )

        m = build_splits(
            tiered_dir=tiered,
            out_dir=tmp_path / "splits",
            bidirectional_dir=bidirectional,
            promote_bidirectional_consistent=True,
            tier_floor="HIGH",
        )
        # LOW + consistent is NOT auto-promoted — bidirectional alone isn't
        # enough to override model-side dissensus.
        assert m.train.count == 0

    def test_inconsistent_med_not_promoted(self, tmp_path: Path) -> None:
        tiered = tmp_path / "tiered"
        bidirectional = tmp_path / "bidirectional"
        _write_tiered(
            tiered,
            "bdsg__pdpa_sg",
            [_make_pair("med-1", "de", "bdsg", "MED", agreement_score=0.75)],
        )
        _write_bidirectional(
            bidirectional,
            "bdsg__pdpa_sg",
            [{"source_id": "med-1", "status": "inconsistent"}],
        )

        m = build_splits(
            tiered_dir=tiered,
            out_dir=tmp_path / "splits",
            bidirectional_dir=bidirectional,
            promote_bidirectional_consistent=True,
        )
        assert m.train.count == 0

    def test_rag_concurs_med_promoted_when_flag_on(self, tmp_path: Path) -> None:
        """MED + rag_concurs=True + promote_rag_concurs=True → in gold."""
        tiered = tmp_path / "tiered"
        rows = [
            _make_pair(
                "med-1",
                "de",
                "bdsg",
                "MED",
                agreement_score=0.75,
                rag_concurs=True,
                rag_vote_citation_id="24",
            ),
        ]
        _write_tiered(tiered, "bdsg__pdpa_sg", rows)
        m = build_splits(
            tiered_dir=tiered,
            out_dir=tmp_path / "splits",
            promote_rag_concurs=True,
        )
        assert m.train.count == 1
        assert m.promote_rag_concurs is True

    def test_rag_concurs_med_NOT_promoted_when_flag_off(self, tmp_path: Path) -> None:
        """Same row, flag off → row not in gold (the flag is opt-in)."""
        tiered = tmp_path / "tiered"
        rows = [
            _make_pair(
                "med-1",
                "de",
                "bdsg",
                "MED",
                agreement_score=0.75,
                rag_concurs=True,
                rag_vote_citation_id="24",
            ),
        ]
        _write_tiered(tiered, "bdsg__pdpa_sg", rows)
        m = build_splits(
            tiered_dir=tiered,
            out_dir=tmp_path / "splits",
            promote_rag_concurs=False,
        )
        assert m.train.count == 0

    def test_rag_concurs_false_med_not_promoted(self, tmp_path: Path) -> None:
        """rag_concurs=False MEDs are not promoted even when flag is on."""
        tiered = tmp_path / "tiered"
        rows = [
            _make_pair(
                "med-1",
                "de",
                "bdsg",
                "MED",
                agreement_score=0.75,
                rag_concurs=False,
            ),
        ]
        _write_tiered(tiered, "bdsg__pdpa_sg", rows)
        m = build_splits(
            tiered_dir=tiered,
            out_dir=tmp_path / "splits",
            promote_rag_concurs=True,
        )
        assert m.train.count == 0

    def test_rag_concurs_does_not_promote_low(self, tmp_path: Path) -> None:
        """Only MED is RAG-promotable; LOW + rag_concurs stays out of gold."""
        tiered = tmp_path / "tiered"
        rows = [
            _make_pair(
                "low-1",
                "de",
                "bdsg",
                "LOW",
                agreement_score=0.5,
                rag_concurs=True,
                rag_vote_citation_id="24",
            ),
        ]
        _write_tiered(tiered, "bdsg__pdpa_sg", rows)
        m = build_splits(
            tiered_dir=tiered,
            out_dir=tmp_path / "splits",
            promote_rag_concurs=True,
        )
        assert m.train.count == 0

    def test_consistent_in_one_pair_does_not_promote_other_pair(self, tmp_path: Path) -> None:
        """Regression: source_id keyspace is per-pair, not global.

        `gdpr-1` is consistent in `gdpr__pdpa_sg` (auto-promote) but
        inconsistent in `gdpr__pdpa_my` (don't promote). Pre-fix code
        keyed the consistent set by source_id only, so a single
        consistent vote leaked across every pair that shared that
        source_id — silently inflating gold by ~4x.
        """
        tiered = tmp_path / "tiered"
        bidirectional = tmp_path / "bidirectional"
        _write_tiered(
            tiered,
            "gdpr__pdpa_sg",
            [
                _make_pair(
                    "gdpr-1",
                    "eu",
                    "gdpr",
                    "MED",
                    target_framework="pdpa_sg",
                    target_jurisdiction="sg",
                    agreement_score=0.75,
                )
            ],
        )
        _write_tiered(
            tiered,
            "gdpr__pdpa_my",
            [
                _make_pair(
                    "gdpr-1",
                    "eu",
                    "gdpr",
                    "MED",
                    target_framework="pdpa_my",
                    target_jurisdiction="my",
                    agreement_score=0.75,
                )
            ],
        )
        _write_bidirectional(
            bidirectional,
            "gdpr__pdpa_sg",
            [{"source_id": "gdpr-1", "status": "consistent"}],
        )
        _write_bidirectional(
            bidirectional,
            "gdpr__pdpa_my",
            [{"source_id": "gdpr-1", "status": "inconsistent"}],
        )

        m = build_splits(
            tiered_dir=tiered,
            out_dir=tmp_path / "splits",
            bidirectional_dir=bidirectional,
            promote_bidirectional_consistent=True,
        )
        # Exactly one promotion (from gdpr__pdpa_sg); my-direction row stays out.
        # eu (source_jurisdiction) lands in train per defaults.
        assert m.train.count == 1
        assert m.train.frameworks == {"gdpr": 1}
