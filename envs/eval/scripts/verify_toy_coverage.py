"""Throwaway helper: verify toy_v1.jsonl coverage against the 2A acceptance criteria.

Run from envs/eval:
    cd envs/eval
    uv run python scripts/verify_toy_coverage.py

Acceptance gates (from docs/development_plan.md §9.2 and data/gold/toy_v1_coverage.md):
    - 20 rows
    - 8 jurisdictions each appearing >=2 (source or target)
    - 'th' >=3 and 'fr' >=3 (native-language moat)
    - Concept axes are documented in toy_v1_coverage.md (not enforced here)
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from daccord.gold import GoldSet

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    gold = GoldSet.from_jsonl(REPO_ROOT / "data" / "gold" / "toy_v1.jsonl")
    n = len(gold.pairs)
    print(f"rows: {n} (gate: 20)")

    jurs = [j for p in gold.pairs for j in (p.source_jurisdiction, p.target_jurisdiction)]
    jur_counts = Counter(jurs)
    print(f"jurisdiction counts (source+target): {dict(jur_counts)}")

    expected = {"eu", "uk", "de", "fr", "sg", "th", "ph", "my"}
    missing = expected - set(jur_counts)
    under = {j: jur_counts.get(j, 0) for j in expected if jur_counts.get(j, 0) < 2}
    print(f"missing jurisdictions: {missing}")
    print(f"under-2 jurisdictions: {under}")

    th, fr = jur_counts.get("th", 0), jur_counts.get("fr", 0)
    print(f"native-language moat: th={th} (gate >=3), fr={fr} (gate >=3)")

    framework_pairs = Counter(p.framework_pair for p in gold.pairs)
    print(f"framework_pair distribution: {dict(framework_pairs)}")

    print(f"\ndataset_hash: {gold.dataset_hash}")

    ok = (n == 20) and not missing and not under and th >= 3 and fr >= 3
    print(f"\nALL ACCEPTANCE GATES: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
