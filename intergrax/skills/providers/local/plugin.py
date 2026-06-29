# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.local.manifests import (
    LOCAL_WORKSPACE_INDEX,
    LOCAL_WORKSPACE_SEARCH,
    LOCAL_WORKSPACE_SYNTHESIZE,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_LOCAL_MANIFESTS = (
    LOCAL_WORKSPACE_INDEX,
    LOCAL_WORKSPACE_SEARCH,
    LOCAL_WORKSPACE_SYNTHESIZE,
)


class LocalSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="local",
            skill_ids=tuple(m.skill_id for m in _LOCAL_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="Local workspace product skills for LKW index, search, and synthesize.",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _LOCAL_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _LOCAL_MANIFESTS:
            registry.register(manifest)
