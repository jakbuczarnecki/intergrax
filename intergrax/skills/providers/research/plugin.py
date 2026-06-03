# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.research.manifests import RESEARCH_LITERATURE_SCAN
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class ResearchSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="research",
            skill_ids=(RESEARCH_LITERATURE_SCAN.skill_id,),
            status=SkillBundleStatus.STABLE,
            description="Research domain skill packs",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (RESEARCH_LITERATURE_SCAN,)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(RESEARCH_LITERATURE_SCAN)
