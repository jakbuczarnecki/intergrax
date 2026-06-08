# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.eval.plugin import EvalSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_eval_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(EvalSkillPlugin, override=override)
