# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.catalog.plugin import CatalogSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_catalog_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(CatalogSkillPlugin, override=override)
