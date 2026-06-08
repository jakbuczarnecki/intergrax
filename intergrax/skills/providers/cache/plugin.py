# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.cache.manifests import (
    CACHE_SESSION_CACHE,
    CACHE_KEY_ADMIN,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_CACHE_MANIFESTS = (
    CACHE_SESSION_CACHE,
    CACHE_KEY_ADMIN,
)


class CacheSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="cache",
            skill_ids=tuple(m.skill_id for m in _CACHE_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="Key-value cache skill packs (SK-EXP2)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _CACHE_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _CACHE_MANIFESTS:
            registry.register(manifest)
