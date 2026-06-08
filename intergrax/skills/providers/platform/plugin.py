# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.platform.manifests import PLATFORM_CONCIERGE
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class PlatformSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="platform",
            skill_ids=(PLATFORM_CONCIERGE.skill_id,),
            status=SkillBundleStatus.STABLE,
            description="Platform hub skill packs (SK-EXP P2)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (PLATFORM_CONCIERGE,)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(PLATFORM_CONCIERGE)
