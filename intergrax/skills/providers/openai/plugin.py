# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.openai.manifests import OPENAI_VECTOR_ADMIN
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class OpenaiSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="openai",
            skill_ids=(OPENAI_VECTOR_ADMIN.skill_id),
            status=SkillBundleStatus.STABLE,
            description="Openai skill packs (SK-EXP4)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (OPENAI_VECTOR_ADMIN)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(OPENAI_VECTOR_ADMIN)
