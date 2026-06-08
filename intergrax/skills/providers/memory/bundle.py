# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.memory.plugin import MemorySkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_memory_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(MemorySkillPlugin, override=override)
