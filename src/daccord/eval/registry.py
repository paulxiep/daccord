"""Registry loader for tier 6A ensemble prompt + tier 7A runner.

The citation registry is the source of truth for "valid citation_ids" per
target framework. Frozen at M1 (tier 5) on disk at
`data/registry/{framework}.json`. Tier 6A stuffs the registry into the
ensemble user message so the four ensemble models can't hallucinate
citation IDs — they must pick from the registered set or return empty.

`load_registry` is process-cached: the 9 framework JSONs total ~1100
citation IDs, so re-reading on every prompt build is wasteful but harmless.
The cache is keyed by absolute path, so a test fixture pointing at a
different `registry_dir` doesn't collide with the production cache.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from daccord.validation import ValidatedModel, validated


class Registry(ValidatedModel):
    """One framework's citation_id list, loaded from data/registry/.

    The on-disk JSON has additional bookkeeping fields (recall stats, source
    SHAs); only the three fields below are load-bearing for tier 6A/7A.
    Pydantic ignores extras by default — extra fields on disk are tolerated.
    """

    framework: str
    jurisdiction: str
    citation_ids: list[str]


@validated
def load_registry(framework: str, registry_dir: Path) -> Registry:
    """Load `data/registry/{framework}.json` and return its citation_ids.

    Raises FileNotFoundError at the system boundary if the framework has
    no registry on disk (e.g. typo in framework name).
    """
    path = registry_dir / f"{framework}.json"
    return _load_registry_cached(str(path.resolve()))


@lru_cache(maxsize=64)
def _load_registry_cached(path_str: str) -> Registry:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Registry not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return Registry.model_validate(data)
