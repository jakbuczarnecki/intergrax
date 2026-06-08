# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.data.plugin import DataSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_data_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(DataSkillPlugin, override=override)
