# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.cache.plugin import CacheSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_cache_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(CacheSkillPlugin, override=override)
