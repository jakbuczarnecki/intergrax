# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.http.plugin import HttpSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_http_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(HttpSkillPlugin, override=override)
