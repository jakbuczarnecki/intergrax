# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.ops.plugin import OpsSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_ops_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(OpsSkillPlugin, override=override)
