# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.skills.registry.bootstrap import register_default_skills, reset_default_skills_for_tests
from intergrax.skills.registry.catalog import clear_skill_catalog
from intergrax.skills.registry.profile import SkillProfile
from intergrax.skills.registry import build_registry_from_profile


@pytest.mark.unit
def test_build_registry_from_profile_legal_bundle() -> None:
    reset_default_skills_for_tests()
    clear_skill_catalog()
    register_default_skills()
    registry = build_registry_from_profile(SkillProfile(enabled_bundles=["legal"]))
    assert registry.has("legal.contract_review")
