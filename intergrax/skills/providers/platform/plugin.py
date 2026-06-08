# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.platform.manifests import (
    PLATFORM_CONCIERGE,
    PLATFORM_SECRETS_FLAGS,
    PLATFORM_CICD_INSPECTOR,
    PLATFORM_SECRET_ADMIN,
    PLATFORM_WORKFLOW_CANCEL,
    PLATFORM_RUNBOOK_HUB,
    PLATFORM_FLAG_ROLLOUT,
    PLATFORM_DEPLOY_INSPECTOR,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_PLATFORM_MANIFESTS = (
    PLATFORM_CONCIERGE,
    PLATFORM_SECRETS_FLAGS,
    PLATFORM_CICD_INSPECTOR,
    PLATFORM_SECRET_ADMIN,
    PLATFORM_WORKFLOW_CANCEL,
    PLATFORM_RUNBOOK_HUB,
    PLATFORM_FLAG_ROLLOUT,
    PLATFORM_DEPLOY_INSPECTOR,
)


class PlatformSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="platform",
            skill_ids=tuple(m.skill_id for m in _PLATFORM_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="platform skill packs (SK-EXP5)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _PLATFORM_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _PLATFORM_MANIFESTS:
            registry.register(manifest)
