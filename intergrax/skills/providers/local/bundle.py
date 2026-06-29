# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.local.plugin import LocalSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_local_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(LocalSkillPlugin, override=override)
