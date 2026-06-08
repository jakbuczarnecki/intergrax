# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.memory.manifests import (
    MEMORY_LTM_CURATOR,
    MEMORY_SESSION_CLEANUP,
    MEMORY_TASK_SCRATCHPAD,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_MEMORY_MANIFESTS = (MEMORY_TASK_SCRATCHPAD, MEMORY_SESSION_CLEANUP, MEMORY_LTM_CURATOR)


class MemorySkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="memory",
            skill_ids=tuple(m.skill_id for m in _MEMORY_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="Task memory skill packs",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _MEMORY_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _MEMORY_MANIFESTS:
            registry.register(manifest)
