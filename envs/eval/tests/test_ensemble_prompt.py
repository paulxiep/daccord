"""Tier 6A ensemble prompt + registry-loader regression guards.

The ensemble prompt is the registry-constrained sibling of `build_eval_prompt`.
Same task shape, plus the citation_id must be a literal string from the target
framework's registered citation_ids. Tests pin the load-bearing constraints so
a non-deliberate edit fails CI before it ships to a $3 batch run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from daccord.eval.prompts import (
    ENSEMBLE_SYSTEM,
    ENSEMBLE_USER_TEMPLATE,
    build_ensemble_prompt,
)
from daccord.eval.registry import Registry, load_registry


@pytest.fixture
def tmp_registry_dir(tmp_path: Path) -> Path:
    """Two fake-framework registries — keeps tests self-contained, no
    dependency on data/registry/ being present in the test environment."""
    (tmp_path / "fakefx_a.json").write_text(
        json.dumps(
            {
                "framework": "fakefx_a",
                "jurisdiction": "xa",
                "citation_ids": ["1", "1A", "2", "3(1)", "3(1)(a)"],
                "extra_field": "tolerated",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "fakefx_b.json").write_text(
        json.dumps(
            {
                "framework": "fakefx_b",
                "jurisdiction": "xb",
                "citation_ids": [f"Section {i}" for i in range(1, 11)],
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


class TestRegistryLoader:
    def test_loads_known_framework(self, tmp_registry_dir: Path) -> None:
        reg = load_registry("fakefx_a", tmp_registry_dir)
        assert isinstance(reg, Registry)
        assert reg.framework == "fakefx_a"
        assert reg.jurisdiction == "xa"
        assert reg.citation_ids == ["1", "1A", "2", "3(1)", "3(1)(a)"]

    def test_tolerates_extra_fields_on_disk(self, tmp_registry_dir: Path) -> None:
        # fakefx_a has an `extra_field` that's not in Registry — must not raise
        reg = load_registry("fakefx_a", tmp_registry_dir)
        assert reg.framework == "fakefx_a"

    def test_raises_on_missing_framework(self, tmp_registry_dir: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_registry("does_not_exist", tmp_registry_dir)

    def test_cached_per_path(self, tmp_registry_dir: Path) -> None:
        # Two calls with the same path should return the SAME object (cache hit).
        r1 = load_registry("fakefx_b", tmp_registry_dir)
        r2 = load_registry("fakefx_b", tmp_registry_dir)
        assert r1 is r2


class TestEnsembleSystemPrompt:
    def test_pins_json_schema_fields(self) -> None:
        for field in ("citation_id", "target_mechanism", "mapping_justification"):
            assert field in ENSEMBLE_SYSTEM

    def test_pins_strict_constraint_language(self) -> None:
        # The registry-pin is the whole point — any prompt edit that
        # weakens this language is a 4-model-batch regression.
        assert "STRICT CONSTRAINT" in ENSEMBLE_SYSTEM
        assert "MUST be one of the exact" in ENSEMBLE_SYSTEM
        assert "Do not invent" in ENSEMBLE_SYSTEM
        assert "Do not paraphrase" in ENSEMBLE_SYSTEM

    def test_pins_no_analogue_escape_hatch(self) -> None:
        # Model must have a way out for true non-mappings (e.g. PDPA-SG has
        # no analog to GDPR Art. 17 right-to-erasure).
        assert "no analogous clause" in ENSEMBLE_SYSTEM
        assert "empty string" in ENSEMBLE_SYSTEM

    def test_pins_no_prefix_normalization(self) -> None:
        # Registry IDs are bare ("1", "32(1)(a)") — adding "Article" or
        # "Section" would break tier 8 exact-match agreement.
        assert "Do not add prefixes" in ENSEMBLE_SYSTEM


class TestEnsembleUserTemplate:
    def test_includes_all_source_target_fields(self) -> None:
        for placeholder in (
            "{source_jurisdiction}",
            "{source_framework}",
            "{source_citation_id}",
            "{source_mechanism}",
            "{target_jurisdiction}",
            "{target_framework}",
            "{registry_block}",
            "{registry_count}",
        ):
            assert placeholder in ENSEMBLE_USER_TEMPLATE

    def test_omits_target_answer_fields(self) -> None:
        # No gold answer exists at tier 7A — and even if one did, leaking
        # it would defeat the purpose of generating candidates.
        assert "{target_citation_id}" not in ENSEMBLE_USER_TEMPLATE
        assert "{target_mechanism}" not in ENSEMBLE_USER_TEMPLATE
        assert "{expected_" not in ENSEMBLE_USER_TEMPLATE


class TestBuildEnsemblePrompt:
    def test_renders_with_small_registry(self) -> None:
        msgs = build_ensemble_prompt(
            source_jurisdiction="eu",
            source_framework="gdpr",
            source_citation_id="Art. 32",
            source_mechanism="Security of processing; appropriate technical measures.",
            target_jurisdiction="sg",
            target_framework="pdpa_sg",
            target_registry=["24", "25", "26"],
        )
        assert msgs.system == ENSEMBLE_SYSTEM
        # Source fields appear verbatim
        assert "Art. 32" in msgs.user
        assert "Security of processing" in msgs.user
        # Target framework appears
        assert "pdpa_sg" in msgs.user
        # All three registry IDs appear quoted in the registry block
        assert '"24"' in msgs.user
        assert '"25"' in msgs.user
        assert '"26"' in msgs.user
        # Registry count is rendered
        assert "3 total" in msgs.user

    def test_renders_passthrough_fields_for_retrieval_client(self) -> None:
        # PromptMessages.source_clause_text + target_jurisdiction are
        # consumed by RetrievalClient at tier 12B; populated here to keep
        # parity with build_eval_prompt.
        msgs = build_ensemble_prompt(
            source_jurisdiction="eu",
            source_framework="gdpr",
            source_citation_id="Art. 32",
            source_mechanism="Security of processing.",
            target_jurisdiction="sg",
            target_framework="pdpa_sg",
            target_registry=["24"],
        )
        assert msgs.source_clause_text == "Security of processing."
        assert msgs.target_jurisdiction == "sg"

    def test_renders_with_large_registry(self) -> None:
        # Sanity: a 288-entry registry (DPA-2018-sized) renders without
        # blowing up and contains both ends of the list.
        ids = [str(i) for i in range(1, 289)]
        msgs = build_ensemble_prompt(
            source_jurisdiction="eu",
            source_framework="gdpr",
            source_citation_id="Art. 32",
            source_mechanism="Security.",
            target_jurisdiction="uk",
            target_framework="dpa_2018",
            target_registry=ids,
        )
        assert '"1"' in msgs.user
        assert '"288"' in msgs.user
        assert "288 total" in msgs.user

    def test_thai_citation_ids_render_intact(self) -> None:
        # Multilingual citation IDs (Thai PDPA uses "มาตรา N") must survive
        # the .format() round-trip without mojibake.
        ids = ["มาตรา 19", "มาตรา 37(4)", "มาตรา 30"]
        msgs = build_ensemble_prompt(
            source_jurisdiction="eu",
            source_framework="gdpr",
            source_citation_id="Art. 6",
            source_mechanism="Lawfulness of processing.",
            target_jurisdiction="th",
            target_framework="pdpa_th",
            target_registry=ids,
        )
        for cid in ids:
            assert f'"{cid}"' in msgs.user

    def test_rejects_positional_args(self) -> None:
        # Keyword-only enforcement: 7 string args is a foot-gun positionally.
        # pydantic's validate_call raises ValidationError, not TypeError.
        with pytest.raises((TypeError, ValidationError)):
            build_ensemble_prompt(  # type: ignore[misc]
                "eu",
                "gdpr",
                "Art. 32",
                "Security.",
                "sg",
                "pdpa_sg",
                ["24"],
            )

    def test_integration_with_real_registry(self) -> None:
        # If the repo's data/registry/ is present (it should be, post-M1),
        # smoke-test that load_registry + build_ensemble_prompt compose.
        repo_root = Path(__file__).resolve().parents[3]
        registry_dir = repo_root / "data" / "registry"
        if not (registry_dir / "gdpr.json").exists():
            pytest.skip("data/registry/gdpr.json not present in this env")
        reg = load_registry("pdpa_sg", registry_dir)
        msgs = build_ensemble_prompt(
            source_jurisdiction="eu",
            source_framework="gdpr",
            source_citation_id="Art. 32",
            source_mechanism="Security of processing.",
            target_jurisdiction="sg",
            target_framework="pdpa_sg",
            target_registry=reg.citation_ids,
        )
        assert reg.framework == "pdpa_sg"
        assert len(reg.citation_ids) > 50  # PDPA-SG registry has ~91 entries
        # First and last registry entries should both appear in the rendered prompt
        assert f'"{reg.citation_ids[0]}"' in msgs.user
        assert f'"{reg.citation_ids[-1]}"' in msgs.user
