# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.storage.manifests import STORAGE_ARTIFACT_SYNC
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class StorageSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="storage",
            skill_ids=(STORAGE_ARTIFACT_SYNC.skill_id,),
            status=SkillBundleStatus.STABLE,
            description="Object storage skill packs (SK-EXP2)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (STORAGE_ARTIFACT_SYNC,)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(STORAGE_ARTIFACT_SYNC)
