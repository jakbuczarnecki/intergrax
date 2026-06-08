# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.identity.plugin import IdentitySkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_identity_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(IdentitySkillPlugin, override=override)
