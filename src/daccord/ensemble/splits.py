"""Tier-7B jurisdiction-disjoint train/val/test splits.

Reads `data/ensemble/tiered/*.jsonl` (tier-6B output), with two optional
overlays:

  - `data/ensemble/validated/*.jsonl` — tier-7C reviewer verdicts.
    Confirms (or rejects) individual MED/LOW/SALVAGE rows.
  - `data/ensemble/bidirectional/*.jsonl` — tier-6B+ cross-direction
    check (output of `scripts/cross_check_ensemble.py`). When
    `promote_bidirectional_consistent=True`, MED rows with
    `status="consistent"` are auto-promoted to gold WITHOUT needing a
    reviewer verdict (the bidirectional agreement IS the verification).

Partitions by `source_jurisdiction` and writes:

  - `data/splits/{train,val,test}.jsonl` — one `TieredPair` per row
  - `data/splits/splits_manifest.json` — counts + jurisdictions + SHA256
    of every input file tree, for MLflow reproducibility

Gold-eligibility rules:

  - `tier == "HIGH"`: always eligible. An overlay row with
    `chosen_citation_id == ""` (reviewer confirmed no-analog) overrides
    this and excludes the row.
  - `tier in {MED, LOW, SALVAGE}`: eligible iff the validated overlay has
    a row for that `source_id` with a non-empty `chosen_citation_id`. An
    empty `chosen_citation_id` means the reviewer confirmed "no analog".
  - MED rows additionally: gold-eligible if bidirectional overlay marks
    them `consistent` AND `promote_bidirectional_consistent=True`.

Jurisdiction disjointness: every `source_jurisdiction` lands in exactly
one of train/val/test (no leakage). Defaults pick the SEA-native
jurisdictions for test (the strongest out-of-domain signal).

Topic disjointness (dev-plan §5 line 195: "breach notification in test,
DSR in train") is NOT implemented in this MR — the topic isn't tagged
in the data. Documented as `topic_disjoint=false` in the manifest.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from daccord.ensemble.schema import Tier, TieredPair
from daccord.validation import ValidatedModel, validated

SplitName = Literal["train", "val", "test"]

DEFAULT_VAL_JURISDICTIONS: tuple[str, ...] = ("my",)
DEFAULT_TEST_JURISDICTIONS: tuple[str, ...] = ("th", "ph")

# Tier-floor inclusion order: HIGH ⊂ MED ⊂ LOW (each floor includes the
# higher-confidence buckets above it). SALVAGE is only included when
# validated; the floor doesn't gate it independently.
_TIER_RANK: dict[Tier, int] = {"HIGH": 3, "MED": 2, "LOW": 1, "SALVAGE": 0}


class SplitSummary(ValidatedModel):
    count: int
    source_jurisdictions: list[str]
    frameworks: dict[str, int]


class SplitsManifest(ValidatedModel):
    created_at: str
    tiered_input_sha256: str
    validated_input_sha256: str | None
    bidirectional_input_sha256: str | None
    promote_bidirectional_consistent: bool
    promote_rag_concurs: bool
    tier_floor: Tier
    val_jurisdictions: list[str]
    test_jurisdictions: list[str]
    topic_disjoint: bool
    topic_disjoint_todo: str
    train: SplitSummary
    val: SplitSummary
    test: SplitSummary


@validated
def build_splits(
    tiered_dir: Path,
    out_dir: Path,
    *,
    validated_dir: Path | None = None,
    bidirectional_dir: Path | None = None,
    promote_bidirectional_consistent: bool = False,
    promote_rag_concurs: bool = False,
    tier_floor: Tier = "HIGH",
    val_jurisdictions: list[str] | None = None,
    test_jurisdictions: list[str] | None = None,
    write: bool = True,
) -> SplitsManifest:
    """Build jurisdiction-disjoint splits from tiered output.

    When `write=False`, returns the manifest without touching the
    filesystem (useful for `--dry-run` and tests).
    """
    val_js = list(val_jurisdictions or DEFAULT_VAL_JURISDICTIONS)
    test_js = list(test_jurisdictions or DEFAULT_TEST_JURISDICTIONS)
    if set(val_js) & set(test_js):
        raise ValueError(
            f"val and test jurisdictions overlap: {sorted(set(val_js) & set(test_js))}"
        )

    tiered_rows, tiered_sha = _load_tiered_rows(tiered_dir)
    validated_map, validated_sha = _load_validated_overlay(validated_dir)
    bidirectional_consistent_ids, bidirectional_sha = _load_bidirectional_consistent_ids(
        bidirectional_dir
    )

    eligible = [
        r
        for r in tiered_rows
        if _is_gold_eligible(
            r,
            tier_floor,
            validated_map,
            bidirectional_consistent_ids if promote_bidirectional_consistent else set(),
            promote_rag_concurs=promote_rag_concurs,
        )
    ]

    by_split: dict[SplitName, list[TieredPair]] = {"train": [], "val": [], "test": []}
    val_set = set(val_js)
    test_set = set(test_js)
    for r in eligible:
        if r.source_jurisdiction in test_set:
            by_split["test"].append(r)
        elif r.source_jurisdiction in val_set:
            by_split["val"].append(r)
        else:
            by_split["train"].append(r)

    manifest = SplitsManifest(
        created_at=datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        tiered_input_sha256=tiered_sha,
        validated_input_sha256=validated_sha,
        bidirectional_input_sha256=bidirectional_sha,
        promote_bidirectional_consistent=promote_bidirectional_consistent,
        promote_rag_concurs=promote_rag_concurs,
        tier_floor=tier_floor,
        val_jurisdictions=sorted(val_js),
        test_jurisdictions=sorted(test_js),
        topic_disjoint=False,
        topic_disjoint_todo=(
            "dev-plan §5 line 195 — defer to future MR with explicit topic tagging"
        ),
        train=_summarize(by_split["train"]),
        val=_summarize(by_split["val"]),
        test=_summarize(by_split["test"]),
    )

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        for split_name, rows in by_split.items():
            _write_split(out_dir / f"{split_name}.jsonl", rows)
        (out_dir / "splits_manifest.json").write_text(
            json.dumps(manifest.model_dump(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return manifest


def _load_tiered_rows(tiered_dir: Path) -> tuple[list[TieredPair], str]:
    files = sorted(tiered_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No tiered files at {tiered_dir}/*.jsonl")
    rows: list[TieredPair] = []
    hasher = hashlib.sha256()
    for path in files:
        data = path.read_bytes()
        hasher.update(path.name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(data)
        for lineno, raw in enumerate(data.decode("utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(TieredPair.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{path}:{lineno}: invalid TieredPair row: {exc}") from exc
    return rows, hasher.hexdigest()


def _load_bidirectional_consistent_ids(
    bidirectional_dir: Path | None,
) -> tuple[set[tuple[str, str]], str | None]:
    """Return `{(forward_pair, source_id)}` for rows with `status="consistent"` + SHA256.

    The composite key is essential: a `source_id` like `gdpr-1` appears in
    multiple forward pairs (`gdpr__pdpa_sg`, `gdpr__pdpa_my`, ...) — being
    consistent in one pair must not auto-promote it in another.

    Hashes every file (not just consistent rows) so the manifest captures
    the full input state for reproducibility.
    """
    if bidirectional_dir is None or not bidirectional_dir.exists():
        return set(), None
    files = sorted(bidirectional_dir.glob("*.jsonl"))
    if not files:
        return set(), None
    consistent: set[tuple[str, str]] = set()
    hasher = hashlib.sha256()
    for path in files:
        forward_pair = path.stem
        data = path.read_bytes()
        hasher.update(path.name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(data)
        for lineno, raw in enumerate(data.decode("utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            sid = row.get("source_id")
            status = row.get("status")
            if not isinstance(sid, str):
                raise ValueError(f"{path}:{lineno}: missing/invalid source_id")
            if status == "consistent":
                consistent.add((forward_pair, sid))
    return consistent, hasher.hexdigest()


def _load_validated_overlay(
    validated_dir: Path | None,
) -> tuple[dict[str, dict[str, object]], str | None]:
    """Return `{source_id: {chosen_citation_id, human_note, ...}}` + SHA256.

    The overlay schema lives in `daccord.ensemble.validated` (tier-7C
    labeler output). Splits only needs `source_id` + `chosen_citation_id`;
    other fields are passed through for reproducibility.
    """
    if validated_dir is None or not validated_dir.exists():
        return {}, None
    files = sorted(validated_dir.glob("*.jsonl"))
    if not files:
        return {}, None
    overlay: dict[str, dict[str, object]] = {}
    hasher = hashlib.sha256()
    for path in files:
        data = path.read_bytes()
        hasher.update(path.name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(data)
        for lineno, raw in enumerate(data.decode("utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            sid = row.get("source_id")
            if not isinstance(sid, str):
                raise ValueError(f"{path}:{lineno}: missing/invalid source_id")
            overlay[sid] = row
    return overlay, hasher.hexdigest()


def _is_gold_eligible(
    row: TieredPair,
    tier_floor: Tier,
    validated_map: dict[str, dict[str, object]],
    bidirectional_consistent_ids: set[tuple[str, str]],
    *,
    promote_rag_concurs: bool = False,
) -> bool:
    """Decide whether a tiered row makes it into the gold pool.

    - HIGH: eligible by default; an overlay row with empty
      `chosen_citation_id` (reviewer confirmed no-analog) overrides this
      and excludes the row.
    - MED: eligible iff (a) bidirectionally consistent in THIS forward pair
      (when caller passed a non-empty `bidirectional_consistent_ids` set),
      OR (b) `rag_concurs=True` (when caller set `promote_rag_concurs`),
      OR (c) tier rank meets `tier_floor` AND an overlay row confirms it.
    - LOW/SALVAGE: eligible iff tier rank meets `tier_floor` AND an
      overlay row confirms it. (Bidirectional + RAG auto-promotion do NOT
      apply to LOW — the secondary signals aren't strong enough to
      override LLM dissensus.)
    """
    validation = validated_map.get(row.source_id)
    if row.tier == "HIGH":
        if validation:
            chosen = validation.get("chosen_citation_id")
            if isinstance(chosen, str) and not chosen:
                return False
        return True
    # MED auto-promotion paths. Composite key (forward_pair, source_id)
    # mandatory because gdpr-1 in gdpr__pdpa_sg is a different cell from
    # gdpr-1 in gdpr__pdpa_my (per-pair leak fix).
    forward_pair = f"{row.source_framework}__{row.target_framework}"
    if row.tier == "MED" and (forward_pair, row.source_id) in bidirectional_consistent_ids:
        return True
    if row.tier == "MED" and promote_rag_concurs and row.rag_concurs:
        return True
    if _TIER_RANK[row.tier] < _TIER_RANK[tier_floor]:
        return False
    if not validation:
        return False
    chosen = validation.get("chosen_citation_id")
    return isinstance(chosen, str) and bool(chosen)


def _summarize(rows: list[TieredPair]) -> SplitSummary:
    return SplitSummary(
        count=len(rows),
        source_jurisdictions=sorted({r.source_jurisdiction for r in rows}),
        frameworks=dict(Counter(r.source_framework for r in rows)),
    )


def _write_split(path: Path, rows: list[TieredPair]) -> None:
    sorted_rows = sorted(rows, key=lambda r: r.source_id)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in sorted_rows:
            f.write(row.model_dump_json())
            f.write("\n")
    tmp.replace(path)
