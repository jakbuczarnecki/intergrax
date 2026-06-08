# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.collaboration.plugin import CollaborationSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_collaboration_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(CollaborationSkillPlugin, override=override)
