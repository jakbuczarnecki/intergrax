# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.context.manifests import CONTEXT_TOKEN_PLANNER
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class ContextSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="context",
            skill_ids=(CONTEXT_TOKEN_PLANNER.skill_id,),
            status=SkillBundleStatus.STABLE,
            description="Context engineering skill packs (SK-EXP3)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (CONTEXT_TOKEN_PLANNER,)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(CONTEXT_TOKEN_PLANNER)
