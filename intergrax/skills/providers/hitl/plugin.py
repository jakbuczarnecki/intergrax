# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.hitl.manifests import (
    HITL_APPROVAL_GATE,
    HITL_QUEUE_MANAGER,
    HITL_ESCALATION_ROUTER,
    HITL_DECISION_AUDITOR,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_HITL_MANIFESTS = (
    HITL_APPROVAL_GATE,
    HITL_QUEUE_MANAGER,
    HITL_ESCALATION_ROUTER,
    HITL_DECISION_AUDITOR,
)


class HitlSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="hitl",
            skill_ids=tuple(m.skill_id for m in _HITL_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="hitl skill packs (SK-EXP5)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _HITL_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _HITL_MANIFESTS:
            registry.register(manifest)
