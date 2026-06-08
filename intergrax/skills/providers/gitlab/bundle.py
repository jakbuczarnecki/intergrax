# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.gitlab.plugin import GitlabSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_gitlab_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(GitlabSkillPlugin, override=override)
