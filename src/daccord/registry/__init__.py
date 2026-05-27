"""Tier-5 citation-registry package.

`schema` defines the on-disk shape of `data/registry/<framework>.json` and
`data/registry/manifest.jsonl`. `patterns` holds per-framework regex
authoring (the language-specific knowledge). `extract` orchestrates a single
framework's extraction from the Tier-4 parsed markdown.

Consumed downstream by:
  - tier 6A: ensemble-prompt JSON schema (citation_id enum / regex constraint
    built from the registry's canonical IDs)
  - tier 6B/8: tiering script SALVAGE detection ("chain-of-thought correct
    but citation_id doesn't exist in the target framework")
  - tier 12C HybridRouter: provenance tagging at serve time
"""

from daccord.registry.schema import FrameworkClauses, FrameworkRegistry, RegistryManifestEntry

__all__ = ["FrameworkClauses", "FrameworkRegistry", "RegistryManifestEntry"]
