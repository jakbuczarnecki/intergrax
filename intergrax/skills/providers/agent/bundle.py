# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.agent.plugin import AgentSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_agent_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(AgentSkillPlugin, override=override)
