# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.health.plugin import HealthSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_health_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(HealthSkillPlugin, override=override)
