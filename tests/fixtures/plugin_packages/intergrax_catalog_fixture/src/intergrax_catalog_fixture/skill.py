# © Artur Czarnecki. All rights reserved.

"""Entry-point skill plugin for catalog fixture tests."""

from __future__ import annotations

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier
from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

FIXTURE_SKILL_ID = "fixture_ep.pack"


class FixturePackSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="fixture_ep",
            skill_ids=(FIXTURE_SKILL_ID,),
            status=SkillBundleStatus.BETA,
            description="Fixture entry-point skill bundle for pytest.",
        )

    @classmethod
    def skill_manifests(cls) -> tuple[SkillManifest, ...]:
        return (
            SkillManifest(
                skill_id=FIXTURE_SKILL_ID,
                version="1.0.0",
                description="Fixture skill referencing fixture_ep.echo tool.",
                tool_ids=("fixture_ep.echo",),
                risk_tier=SkillRiskTier.LOW,
                tags=("fixture",),
            ),
        )

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in cls.skill_manifests():
            registry.register(manifest)
