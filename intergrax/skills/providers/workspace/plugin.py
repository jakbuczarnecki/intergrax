# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.workspace.manifests import WORKSPACE_AUTHORING, WORKSPACE_SNAPSHOT_MANAGER
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class WorkspaceSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="workspace",
            skill_ids=(WORKSPACE_AUTHORING.skill_id, WORKSPACE_SNAPSHOT_MANAGER.skill_id),
            status=SkillBundleStatus.STABLE,
            description="Shadow workspace skill packs (SK-EXP P0 + SK-EXP3)",
        )

    _MANIFESTS = (WORKSPACE_AUTHORING, WORKSPACE_SNAPSHOT_MANAGER)

    @classmethod
    def skill_manifests(cls) -> tuple:
        return WorkspaceSkillPlugin._MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in WorkspaceSkillPlugin._MANIFESTS:
            registry.register(manifest)
