# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.vector_store.manifests import (
    VECTOR_STORE_ADMIN,
    VECTOR_STORE_PURGE,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_VECTOR_STORE_MANIFESTS = (
    VECTOR_STORE_ADMIN,
    VECTOR_STORE_PURGE,
)


class VectorStoreSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="vector_store",
            skill_ids=tuple(m.skill_id for m in _VECTOR_STORE_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="Vector store administration skill packs (SK-EXP3)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _VECTOR_STORE_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _VECTOR_STORE_MANIFESTS:
            registry.register(manifest)
