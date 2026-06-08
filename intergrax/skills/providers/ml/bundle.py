# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.ml.plugin import MlSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_ml_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(MlSkillPlugin, override=override)
