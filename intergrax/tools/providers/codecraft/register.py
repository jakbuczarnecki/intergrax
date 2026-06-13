# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.registry.plugin_register import register_tool_plugin
from intergrax.tools.registry.shipped_plugins import CodeCraftToolPlugin


def register_codecraft_tool_bundle(*, override: bool = False) -> None:
    register_tool_plugin(CodeCraftToolPlugin, override=override)
