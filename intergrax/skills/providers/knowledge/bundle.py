# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.knowledge.plugin import KnowledgeSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_knowledge_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(KnowledgeSkillPlugin, override=override)
