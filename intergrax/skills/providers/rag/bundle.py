# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.rag.plugin import RagSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_rag_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(RagSkillPlugin, override=override)
