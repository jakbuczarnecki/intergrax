# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.research.manifests import RESEARCH_LITERATURE_SCAN
from intergrax.skills.registry.catalog import SkillBundleEntry, SkillBundleStatus, register_skill_bundle
from intergrax.skills.registry.runtime import SkillRegistry


def _register_research_skills(registry: SkillRegistry) -> None:
    registry.register(RESEARCH_LITERATURE_SCAN)


def register_research_skill_bundle(*, override: bool = False) -> None:
    register_skill_bundle(
        SkillBundleEntry(
            bundle_id="research",
            skill_ids=(RESEARCH_LITERATURE_SCAN.skill_id,),
            register=_register_research_skills,
            status=SkillBundleStatus.STABLE,
            description="Research domain skill packs",
        ),
        override=override,
    )
