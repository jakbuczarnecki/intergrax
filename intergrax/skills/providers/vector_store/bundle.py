# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.vector_store.plugin import VectorStoreSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_vector_store_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(VectorStoreSkillPlugin, override=override)
