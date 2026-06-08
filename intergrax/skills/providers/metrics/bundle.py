# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.metrics.plugin import MetricsSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_metrics_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(MetricsSkillPlugin, override=override)
