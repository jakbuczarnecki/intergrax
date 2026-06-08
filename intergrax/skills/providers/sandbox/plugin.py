# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.sandbox.manifests import SANDBOX_CODE_EXEC
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class SandboxSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="sandbox",
            skill_ids=(SANDBOX_CODE_EXEC.skill_id,),
            status=SkillBundleStatus.STABLE,
            description="Sandbox execution skill packs (SK-EXP2)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (SANDBOX_CODE_EXEC,)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(SANDBOX_CODE_EXEC)
