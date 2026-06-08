# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.code.manifests import CODE_RUNNER
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class CodeSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="code",
            skill_ids=(CODE_RUNNER.skill_id),
            status=SkillBundleStatus.STABLE,
            description="Code skill packs (SK-EXP4)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (CODE_RUNNER)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(CODE_RUNNER)
