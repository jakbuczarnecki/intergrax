# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.message_bus.plugin import MessageBusSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_message_bus_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(MessageBusSkillPlugin, override=override)
