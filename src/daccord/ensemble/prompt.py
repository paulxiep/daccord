"""Strategy-agnostic prompt staging for tier-7A ensemble generation.

`BatchPrompt` + `model_slug` live here so every strategy backend (Bedrock,
paid direct API, free-tier ensemble, EC2-hosted) shares the same in-flight
prompt shape and the same on-disk + S3 file naming. Previously these lived
in `src/daccord/aws/batch.py`; that module now re-exports them for
back-compat so existing tests keep passing.

Output JSONL files follow the convention
`data/ensemble/raw/{framework_pair}__{model_slug}.jsonl`, consumed by
`scripts/tier_ensemble.py` (tier 6B).
"""

from __future__ import annotations

import re

from daccord.validation import ValidatedModel, validated


@validated
def model_slug(model_id: str) -> str:
    """Map any model identifier to a filename-safe slug.

    Bedrock model IDs like `meta.llama4-scout-17b-instruct-v1:0` contain `:`
    and `.` which are invalid on Windows and noisy in S3 keys. OpenRouter-style
    IDs like `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` contain `/`.
    All of those get squashed to single dashes; consecutive dashes collapse.

    The slug is the stable identifier for the output file's model component:
    `data/ensemble/raw/{pair}__{model_slug}.jsonl`.
    """
    s = re.sub(r"[:./]", "-", model_id)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


class BatchPrompt(ValidatedModel):
    """One source-clause prompt staged for one ensemble seat.

    `record_id` is a unique-within-batch identifier (we use `source_id`
    directly, which is unique per pair). The source_* fields are carried
    alongside the prompt text so the output parser can stitch the source-side
    context back into `EnsembleCandidate` without re-loading the clauses file.
    """

    record_id: str
    source_id: str
    source_jurisdiction: str
    source_framework: str
    source_citation_id: str
    source_mechanism: str
    target_jurisdiction: str
    target_framework: str
    system: str
    user: str
    max_tokens: int
