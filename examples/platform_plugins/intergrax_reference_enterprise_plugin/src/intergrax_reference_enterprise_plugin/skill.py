# © Artur Czarnecki. All rights reserved.

"""Reference SkillPlugin surface for the enterprise multi-capability package."""

from __future__ import annotations

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier
from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

from intergrax_reference_enterprise_plugin.tool import REFERENCE_ENTERPRISE_ECHO_TOOL_ID

REFERENCE_ENTERPRISE_SKILL_ID = "reference_enterprise.pack"


class ReferenceEnterprisePackSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="reference_enterprise_pack",
            skill_ids=(REFERENCE_ENTERPRISE_SKILL_ID,),
            status=SkillBundleStatus.BETA,
            description="Reference skill bundle referencing reference_enterprise.echo.",
        )

    @classmethod
    def skill_manifests(cls) -> tuple[SkillManifest, ...]:
        return (
            SkillManifest(
                skill_id=REFERENCE_ENTERPRISE_SKILL_ID,
                version="1.0.0",
                description="Reference enterprise skill requiring the bundled echo tool.",
                tool_ids=(REFERENCE_ENTERPRISE_ECHO_TOOL_ID,),
                risk_tier=SkillRiskTier.LOW,
                tags=("reference", "platform-plugin-docs-6"),
            ),
        )

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in cls.skill_manifests():
            registry.register(manifest)
