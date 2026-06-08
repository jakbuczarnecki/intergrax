# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.openai.plugin import OpenaiSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_openai_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(OpenaiSkillPlugin, override=override)
