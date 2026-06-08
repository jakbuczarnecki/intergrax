# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.ops.manifests import (
    OPS_TRACE_DEBUG,
    OPS_INCIDENT_DISPATCH,
    OPS_SECURITY_AUDIT,
    OPS_WORKFLOW_RUNNER,
    OPS_WORKFLOW_ADMIN,
    OPS_FINDINGS_REVIEW,
    OPS_LOG_TAIL,
    OPS_INCIDENT_ACK,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_OPS_MANIFESTS = (
    OPS_TRACE_DEBUG,
    OPS_INCIDENT_DISPATCH,
    OPS_SECURITY_AUDIT,
    OPS_WORKFLOW_RUNNER,
    OPS_WORKFLOW_ADMIN,
    OPS_FINDINGS_REVIEW,
    OPS_LOG_TAIL,
    OPS_INCIDENT_ACK,
)


class OpsSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="ops",
            skill_ids=tuple(m.skill_id for m in _OPS_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="Operations and reliability skill packs (SK-EXP P1)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _OPS_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _OPS_MANIFESTS:
            registry.register(manifest)
