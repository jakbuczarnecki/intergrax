# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.hitl.plugin import HitlSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_hitl_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(HitlSkillPlugin, override=override)
