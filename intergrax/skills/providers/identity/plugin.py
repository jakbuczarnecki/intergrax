# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.identity.manifests import (
    IDENTITY_ACCESS_CHECKER,
    IDENTITY_SESSION_BOOTSTRAP,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_IDENTITY_MANIFESTS = (
    IDENTITY_ACCESS_CHECKER,
    IDENTITY_SESSION_BOOTSTRAP,
)


class IdentitySkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="identity",
            skill_ids=tuple(m.skill_id for m in _IDENTITY_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="identity skill packs (SK-EXP5)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _IDENTITY_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _IDENTITY_MANIFESTS:
            registry.register(manifest)
