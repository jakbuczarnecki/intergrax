# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.jira.plugin import JiraSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_jira_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(JiraSkillPlugin, override=override)
