# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.health.manifests import (
    HEALTH_INTEGRATION_PROBE,
    HEALTH_FULL_STACK_PROBE,
    HEALTH_IDENTITY_PROBE,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_HEALTH_MANIFESTS = (
    HEALTH_INTEGRATION_PROBE,
    HEALTH_FULL_STACK_PROBE,
    HEALTH_IDENTITY_PROBE,
)


class HealthSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="health",
            skill_ids=tuple(m.skill_id for m in _HEALTH_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="health skill packs (SK-EXP5)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _HEALTH_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _HEALTH_MANIFESTS:
            registry.register(manifest)
