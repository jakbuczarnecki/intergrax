# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.workspace.manifests import (
    WORKSPACE_AUTHORING,
    WORKSPACE_SNAPSHOT_MANAGER,
    WORKSPACE_DRAFT_REVIEWER,
    WORKSPACE_ARTIFACT_EXPORTER,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_WORKSPACE_MANIFESTS = (
    WORKSPACE_AUTHORING,
    WORKSPACE_SNAPSHOT_MANAGER,
    WORKSPACE_DRAFT_REVIEWER,
    WORKSPACE_ARTIFACT_EXPORTER,
)


class WorkspaceSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="workspace",
            skill_ids=tuple(m.skill_id for m in _WORKSPACE_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="workspace skill packs (SK-EXP5)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _WORKSPACE_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _WORKSPACE_MANIFESTS:
            registry.register(manifest)
