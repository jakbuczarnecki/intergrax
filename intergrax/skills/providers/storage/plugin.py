# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.storage.manifests import (
    STORAGE_ARTIFACT_SYNC,
    STORAGE_OBJECT_LIFECYCLE,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_STORAGE_MANIFESTS = (
    STORAGE_ARTIFACT_SYNC,
    STORAGE_OBJECT_LIFECYCLE,
)


class StorageSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="storage",
            skill_ids=tuple(m.skill_id for m in _STORAGE_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="Object storage skill packs (SK-EXP2)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _STORAGE_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _STORAGE_MANIFESTS:
            registry.register(manifest)
