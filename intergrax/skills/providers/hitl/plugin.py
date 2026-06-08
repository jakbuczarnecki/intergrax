# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.hitl.manifests import HITL_APPROVAL_GATE
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class HitlSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="hitl",
            skill_ids=(HITL_APPROVAL_GATE.skill_id,),
            status=SkillBundleStatus.STABLE,
            description="Human-in-the-loop governance skill packs (SK-EXP2)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (HITL_APPROVAL_GATE,)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(HITL_APPROVAL_GATE)
