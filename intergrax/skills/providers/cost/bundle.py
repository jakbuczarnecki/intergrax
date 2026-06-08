# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.cost.plugin import CostSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_cost_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(CostSkillPlugin, override=override)
