# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.modality.plugin import ModalitySkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_modality_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(ModalitySkillPlugin, override=override)
