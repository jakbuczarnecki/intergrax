# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.legal.plugin import LegalSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_legal_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(LegalSkillPlugin, override=override)
