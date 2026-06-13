# © Artur Czarnecki. All rights reserved.

import pytest

pytestmark = [pytest.mark.no_ci]

from intergrax.skills.registry.bootstrap import register_default_skills, reset_default_skills_for_tests
from intergrax.skills.registry.catalog import catalog_snapshot, clear_skill_catalog
from intergrax.skills.registry.profile import SkillProfile
from intergrax.skills.registry import build_registry_from_profile


@pytest.mark.unit
def test_build_registry_from_profile_legal_bundle() -> None:
    reset_default_skills_for_tests()
    clear_skill_catalog()
    register_default_skills()
    registry = build_registry_from_profile(SkillProfile(enabled_bundles=["legal"]))
    assert registry.has("legal.contract_review")


@pytest.mark.unit
def test_build_registry_from_profile_research_bundle() -> None:
    reset_default_skills_for_tests()
    clear_skill_catalog()
    register_default_skills()
    registry = build_registry_from_profile(SkillProfile(enabled_bundles=["research"]))
    assert registry.has("research.literature_scan")


@pytest.mark.unit
def test_register_default_skills_includes_legal_and_research() -> None:
    reset_default_skills_for_tests()
    clear_skill_catalog()
    register_default_skills()
    registry = build_registry_from_profile(SkillProfile())
    assert registry.has("legal.contract_review")
    assert registry.has("research.literature_scan")


@pytest.mark.unit
@pytest.mark.gate
def test_register_default_skills_idempotent_after_partial_bootstrap() -> None:
    """CI xdist: partial Tier-3 bootstrap then full register must not raise."""
    from intergrax.core.catalog_bootstrap import bootstrap_catalogs

    reset_default_skills_for_tests()
    bootstrap_catalogs(register_shipped=True, skill_bundle_ids=["harness", "agent"])
    register_default_skills()
    snap = catalog_snapshot()
    assert "harness" in snap
    assert "agent" in snap
    assert "legal" in snap
