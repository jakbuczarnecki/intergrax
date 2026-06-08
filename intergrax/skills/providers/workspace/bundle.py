# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.workspace.plugin import WorkspaceSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_workspace_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(WorkspaceSkillPlugin, override=override)
