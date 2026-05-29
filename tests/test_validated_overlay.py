"""Tier-7C validated-overlay writer guards.

Mirrors the immutability tests in `test_ensemble_strategy.py` against the
overlay layer. A reviewed pair is locked; the only way to re-review is to
delete the file (or manually prune that source_id).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from daccord.ensemble.strategy import ImmutabilityViolation
from daccord.ensemble.validated import (
    ValidatedPair,
    append_validation,
    load_validated_source_ids,
    read_validations,
)


def _make(
    source_id: str = "src-1",
    chosen: str = "24",
    note: str = "",
    reviewer: str = "alice",
) -> ValidatedPair:
    return ValidatedPair(
        source_id=source_id,
        source_jurisdiction="eu",
        source_framework="gdpr",
        target_jurisdiction="sg",
        target_framework="pdpa_sg",
        chosen_citation_id=chosen,
        human_note=note,
        reviewer=reviewer,
        reviewed_at="2026-05-27T12:00:00Z",
        tier_at_review="MED",
        agreement_score_at_review=0.75,
    )


def test_append_then_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    append_validation(path, _make("src-1"))
    append_validation(path, _make("src-2", chosen="", note="no analog"))

    rows = read_validations(path)
    assert {r.source_id for r in rows} == {"src-1", "src-2"}
    assert {r.source_id for r in rows if r.chosen_citation_id == ""} == {"src-2"}


def test_load_validated_source_ids(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    append_validation(path, _make("src-1"))
    append_validation(path, _make("src-2"))
    assert load_validated_source_ids(path) == {"src-1", "src-2"}


def test_load_validated_source_ids_missing_returns_empty(tmp_path: Path) -> None:
    assert load_validated_source_ids(tmp_path / "nope.jsonl") == set()


def test_append_refuses_duplicate_source_id(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    append_validation(path, _make("src-1", chosen="24"))
    with pytest.raises(ImmutabilityViolation, match="duplicate source_id='src-1'"):
        append_validation(path, _make("src-1", chosen="25"))


def test_append_refuses_duplicate_even_for_no_analog(tmp_path: Path) -> None:
    # A "no analog" verdict locks the row too — operator can't flip it
    # without manually deleting + re-appending.
    path = tmp_path / "out.jsonl"
    append_validation(path, _make("src-1", chosen=""))
    with pytest.raises(ImmutabilityViolation):
        append_validation(path, _make("src-1", chosen="24"))


def test_read_validations_missing_returns_empty(tmp_path: Path) -> None:
    assert read_validations(tmp_path / "nope.jsonl") == []


def test_round_trip_preserves_all_fields(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    original = _make(source_id="src-1", chosen="24(1)", note="picked subclause", reviewer="bob")
    append_validation(path, original)
    restored = read_validations(path)[0]
    assert restored == original
