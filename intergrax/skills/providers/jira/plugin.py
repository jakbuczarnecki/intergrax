# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.jira.manifests import JIRA_TASK_NAVIGATOR
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class JiraSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="jira",
            skill_ids=(JIRA_TASK_NAVIGATOR.skill_id),
            status=SkillBundleStatus.STABLE,
            description="Jira skill packs (SK-EXP4)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (JIRA_TASK_NAVIGATOR)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(JIRA_TASK_NAVIGATOR)
