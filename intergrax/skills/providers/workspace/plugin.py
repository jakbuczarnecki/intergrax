# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.workspace.manifests import WORKSPACE_AUTHORING
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class WorkspaceSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="workspace",
            skill_ids=(WORKSPACE_AUTHORING.skill_id,),
            status=SkillBundleStatus.STABLE,
            description="Shadow workspace authoring skill packs (SK-EXP P0)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (WORKSPACE_AUTHORING,)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(WORKSPACE_AUTHORING)
