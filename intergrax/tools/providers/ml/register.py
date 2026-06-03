# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.tools.registry.plugin_register import register_tool_plugin
from intergrax.tools.registry.shipped_plugins import MlToolPlugin


def register_ml_tool_bundle(*, override: bool = False) -> None:
    register_tool_plugin(MlToolPlugin, override=override)
