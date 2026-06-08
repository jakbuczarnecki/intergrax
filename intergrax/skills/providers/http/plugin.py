# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.http.manifests import HTTP_API_CLIENT
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class HttpSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="http",
            skill_ids=(HTTP_API_CLIENT.skill_id),
            status=SkillBundleStatus.STABLE,
            description="Http skill packs (SK-EXP4)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (HTTP_API_CLIENT)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(HTTP_API_CLIENT)
