# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.harness.plugin import HarnessSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_harness_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(HarnessSkillPlugin, override=override)
