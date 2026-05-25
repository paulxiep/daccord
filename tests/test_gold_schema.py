"""Gold-set schema tests — the pipeline-wide data contract.

`GoldPair` is the 2A/2B interface AND the eventual input to tier 7A
(ensemble candidate generation, stratified sampling) and tier 10A
(training-loop trainer dataset). Breaking these tests breaks every
downstream consumer; fix the contract deliberately.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from daccord.gold import GoldPair, GoldSet

TOY_ROW = {
    "id": "tg-001",
    "source_jurisdiction": "eu",
    "source_framework": "gdpr",
    "source_citation_id": "Art. 32",
    "source_mechanism": (
        "Security of processing; appropriate technical and organisational measures."
    ),
    "source_language": "en",
    "target_jurisdiction": "sg",
    "target_framework": "pdpa_sg",
    "target_citation_id": "Section 24",
    "target_mechanism": "Protection of personal data with reasonable security arrangements.",
    "target_language": "en",
    "notes": None,
}


class TestGoldPair:
    def test_round_trip_from_json(self) -> None:
        pair = GoldPair.model_validate(TOY_ROW)
        assert pair.id == "tg-001"
        assert pair.framework_pair == "gdpr__pdpa_sg"
        assert pair.notes is None

    def test_missing_required_field_rejected(self) -> None:
        bad = {k: v for k, v in TOY_ROW.items() if k != "target_citation_id"}
        with pytest.raises(ValidationError):
            GoldPair.model_validate(bad)

    def test_wrong_type_rejected(self) -> None:
        bad = {**TOY_ROW, "source_jurisdiction": 123}
        with pytest.raises(ValidationError):
            GoldPair.model_validate(bad)

    def test_notes_is_optional(self) -> None:
        with_notes = {**TOY_ROW, "notes": "FR/DE require partial-text alignment review"}
        pair = GoldPair.model_validate(with_notes)
        assert pair.notes is not None

    def test_framework_pair_is_underscore_separated(self) -> None:
        pair = GoldPair.model_validate(TOY_ROW)
        assert pair.framework_pair == f"{pair.source_framework}__{pair.target_framework}"


class TestGoldSet:
    def test_load_two_row_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "toy.jsonl"
        row_a = json.dumps(TOY_ROW)
        row_b = json.dumps({**TOY_ROW, "id": "tg-002", "target_jurisdiction": "th"})
        path.write_text(f"{row_a}\n{row_b}\n", encoding="utf-8")

        gold = GoldSet.from_jsonl(path)
        assert len(gold.pairs) == 2
        assert gold.pairs[0].id == "tg-001"
        assert gold.pairs[1].target_jurisdiction == "th"
        assert gold.source_path == path.as_posix()

    def test_dataset_hash_matches_file_sha256(self, tmp_path: Path) -> None:
        path = tmp_path / "toy.jsonl"
        payload = (json.dumps(TOY_ROW) + "\n").encode("utf-8")
        path.write_bytes(payload)
        gold = GoldSet.from_jsonl(path)
        assert gold.dataset_hash == hashlib.sha256(payload).hexdigest()

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "toy.jsonl"
        path.write_text(
            "\n" + json.dumps(TOY_ROW) + "\n\n" + json.dumps({**TOY_ROW, "id": "tg-002"}) + "\n",
            encoding="utf-8",
        )
        gold = GoldSet.from_jsonl(path)
        assert len(gold.pairs) == 2

    def test_invalid_row_raises_with_lineno(self, tmp_path: Path) -> None:
        path = tmp_path / "toy.jsonl"
        path.write_text(
            json.dumps(TOY_ROW)
            + "\n"
            + json.dumps({"id": "broken"})  # missing required fields
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":2:"):
            GoldSet.from_jsonl(path)

    def test_empty_file_yields_empty_gold_set(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        gold = GoldSet.from_jsonl(path)
        assert gold.pairs == []
        assert gold.dataset_hash == hashlib.sha256(b"").hexdigest()
