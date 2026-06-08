# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.crm.plugin import CrmSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_crm_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(CrmSkillPlugin, override=override)
