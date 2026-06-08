# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.gitlab.manifests import GITLAB_ISSUE_CREATOR
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class GitlabSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="gitlab",
            skill_ids=(GITLAB_ISSUE_CREATOR.skill_id),
            status=SkillBundleStatus.STABLE,
            description="Gitlab skill packs (SK-EXP4)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (GITLAB_ISSUE_CREATOR)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(GITLAB_ISSUE_CREATOR)
