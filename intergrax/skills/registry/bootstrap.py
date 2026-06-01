# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.harness.bundle import register_harness_skill_bundle
from intergrax.skills.providers.legal.bundle import register_legal_skill_bundle
from intergrax.skills.providers.research.bundle import register_research_skill_bundle
from intergrax.skills.registry.catalog import register_skill_bundle

_BOOTSTRAPPED = False


def register_default_skills() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    register_harness_skill_bundle()
    register_legal_skill_bundle()
    register_research_skill_bundle()
    _BOOTSTRAPPED = True


def reset_default_skills_for_tests() -> None:
    global _BOOTSTRAPPED
    from intergrax.skills.registry.catalog import clear_skill_catalog

    clear_skill_catalog()
    _BOOTSTRAPPED = False
