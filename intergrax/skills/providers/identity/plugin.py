# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.identity.manifests import IDENTITY_ACCESS_CHECKER
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class IdentitySkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="identity",
            skill_ids=(IDENTITY_ACCESS_CHECKER.skill_id,),
            status=SkillBundleStatus.STABLE,
            description="Identity and access skill packs (SK-EXP3)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (IDENTITY_ACCESS_CHECKER,)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(IDENTITY_ACCESS_CHECKER)
