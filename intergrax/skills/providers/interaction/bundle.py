# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.interaction.plugin import InteractionSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_interaction_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(InteractionSkillPlugin, override=override)
