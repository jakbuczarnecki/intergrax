# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.cloud_platform.manifests import CLOUD_PLATFORM_RESOLVER
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class CloudPlatformSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="cloud_platform",
            skill_ids=(CLOUD_PLATFORM_RESOLVER.skill_id),
            status=SkillBundleStatus.STABLE,
            description="Cloud_Platform skill packs (SK-EXP4)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (CLOUD_PLATFORM_RESOLVER)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(CLOUD_PLATFORM_RESOLVER)
