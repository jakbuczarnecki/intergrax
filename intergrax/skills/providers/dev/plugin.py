# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.dev.manifests import (
    DEV_ISSUE_TRIAGE,
    DEV_ISSUE_CREATOR,
    DEV_ISSUE_UPDATER,
    DEV_PR_REVIEWER,
    DEV_RELEASE_NOTES,
    DEV_SPRINT_PLANNER,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_DEV_MANIFESTS = (
    DEV_ISSUE_TRIAGE,
    DEV_ISSUE_CREATOR,
    DEV_ISSUE_UPDATER,
    DEV_PR_REVIEWER,
    DEV_RELEASE_NOTES,
    DEV_SPRINT_PLANNER,
)


class DevSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="dev",
            skill_ids=tuple(m.skill_id for m in _DEV_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="dev skill packs (SK-EXP5)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _DEV_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _DEV_MANIFESTS:
            registry.register(manifest)
