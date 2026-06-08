# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.cache.manifests import CACHE_SESSION_CACHE
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class CacheSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="cache",
            skill_ids=(CACHE_SESSION_CACHE.skill_id,),
            status=SkillBundleStatus.STABLE,
            description="Key-value cache skill packs (SK-EXP2)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (CACHE_SESSION_CACHE,)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(CACHE_SESSION_CACHE)
