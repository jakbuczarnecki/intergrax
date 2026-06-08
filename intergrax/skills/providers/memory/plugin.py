# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.memory.manifests import MEMORY_TASK_SCRATCHPAD
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class MemorySkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="memory",
            skill_ids=(MEMORY_TASK_SCRATCHPAD.skill_id,),
            status=SkillBundleStatus.STABLE,
            description="Task memory skill packs (SK-EXP P0)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (MEMORY_TASK_SCRATCHPAD,)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(MEMORY_TASK_SCRATCHPAD)
