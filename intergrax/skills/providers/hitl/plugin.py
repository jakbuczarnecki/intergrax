# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.hitl.manifests import HITL_APPROVAL_GATE, HITL_QUEUE_MANAGER
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class HitlSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="hitl",
            skill_ids=(HITL_APPROVAL_GATE.skill_id, HITL_QUEUE_MANAGER.skill_id),
            status=SkillBundleStatus.STABLE,
            description="Human-in-the-loop governance skill packs (SK-EXP2 + SK-EXP3)",
        )

    _MANIFESTS = (HITL_APPROVAL_GATE, HITL_QUEUE_MANAGER)

    @classmethod
    def skill_manifests(cls) -> tuple:
        return HitlSkillPlugin._MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in HitlSkillPlugin._MANIFESTS:
            registry.register(manifest)
