# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.sandbox.plugin import SandboxSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_sandbox_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(SandboxSkillPlugin, override=override)
