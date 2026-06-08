# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.browser.plugin import BrowserSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_browser_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(BrowserSkillPlugin, override=override)
