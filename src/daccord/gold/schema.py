"""Gold-set data contracts — pipeline-spanning, NOT eval-specific.

`GoldPair` is the canonical shape of one hand-validated cross-jurisdiction
mapping row. Consumed by:

  - tier 2A: hand-built toy gold (20 pairs at M0)
  - tier 2B: eval harness input (this is the 2A/2B interface)
  - tier 7A: ensemble candidate generation — stratified sampling by
    jurisdiction / framework_pair
  - tier 9 : gold freeze → train/val/test splits with dataset SHA
  - tier 10A: training loop loads splits[*].jsonl as the trainer dataset

Lives in `daccord.gold` (not `daccord.eval`) so a training script doesn't
have to import its data shape from the eval module.

`GoldSet.from_jsonl` is the canonical loader — every consumer goes through
it so the SHA256-of-the-file (used as `dataset_hash` in MLflow params)
stays consistent across tiers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

from daccord.tracking import compute_file_sha256
from daccord.validation import ValidatedModel


class GoldPair(ValidatedModel):
    """One hand-validated cross-jurisdiction mapping row.

    Field semantics:
      - `source_*`: the *query* — the regulatory clause we ask the model
        about. `source_citation_id` is shown to the model verbatim.
      - `target_*`: the *gold answer* — what the model should return.
        `target_citation_id` is the Tier-1 exact-match target.
      - `*_language`: ISO 639-1 codes, used for the per-language metric
        breakdown that quantifies the SEA/FR/DE differentiation
        (development_plan.md §5).
      - `notes`: author commentary, not consumed by the harness.
    """

    id: str
    source_jurisdiction: str
    source_framework: str
    source_citation_id: str
    source_mechanism: str
    source_language: str
    target_jurisdiction: str
    target_framework: str
    target_citation_id: str
    target_mechanism: str
    target_language: str
    notes: str | None = None

    @property
    def framework_pair(self) -> str:
        """Stable key for per-framework-pair aggregation in the eval CSV."""
        return f"{self.source_framework}__{self.target_framework}"


class GoldSet(ValidatedModel):
    """In-memory view of a gold JSONL file plus its SHA256 for MLflow."""

    pairs: list[GoldPair]
    dataset_hash: str
    source_path: str

    @classmethod
    def from_jsonl(cls, path: Path) -> Self:
        """Load a JSONL file of GoldPair rows; hash the file for the contract.

        Skips blank lines so hand-edited gold files tolerate trailing newlines.
        Validation runs per-row; a malformed row raises with its line number
        so the author knows where to look.
        """
        pairs: list[GoldPair] = []
        text = path.read_text(encoding="utf-8")
        for lineno, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                pairs.append(GoldPair.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{path}:{lineno}: invalid GoldPair row: {exc}") from exc
        return cls(
            pairs=pairs,
            dataset_hash=compute_file_sha256(path),
            source_path=str(path.as_posix()),
        )
