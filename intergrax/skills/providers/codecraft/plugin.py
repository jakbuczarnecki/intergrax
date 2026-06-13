# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.codecraft.manifests import CODECRAFT_EPHEMERAL_BUILDER
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class CodeCraftSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="codecraft",
            skill_ids=(CODECRAFT_EPHEMERAL_BUILDER.skill_id,),
            status=SkillBundleStatus.STABLE,
            description="Ephemeral Code Craft skill packs (ECC-2.7).",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (CODECRAFT_EPHEMERAL_BUILDER,)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(CODECRAFT_EPHEMERAL_BUILDER)


def register_codecraft_skill_bundle(*, override: bool = False) -> None:
    from intergrax.skills.registry.plugin_register import register_skill_plugin

    register_skill_plugin(CodeCraftSkillPlugin, override=override)
