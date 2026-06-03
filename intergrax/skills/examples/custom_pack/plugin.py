# © Artur Czarnecki. All rights reserved.

"""Reference :class:`SkillPlugin` for external skill packages."""

from __future__ import annotations

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier
from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

CUSTOM_PACK_SKILL_ID = "custom_pack.demo"


class CustomPackSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="custom_pack",
            skill_ids=(CUSTOM_PACK_SKILL_ID,),
            status=SkillBundleStatus.BETA,
            description="Example external skill bundle.",
        )

    @classmethod
    def skill_manifests(cls) -> tuple[SkillManifest, ...]:
        return (
            SkillManifest(
                skill_id=CUSTOM_PACK_SKILL_ID,
                version="1.0.0",
                description="Example skill pack for external authors.",
                tool_ids=("custom_echo.ping",),
                risk_tier=SkillRiskTier.LOW,
                tags=("example",),
            ),
        )

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in cls.skill_manifests():
            registry.register(manifest)
