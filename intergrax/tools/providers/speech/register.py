# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.tools.registry.plugin_register import register_tool_plugin
from intergrax.tools.registry.shipped_plugins import SpeechToolPlugin


def register_speech_tool_bundle(*, override: bool = False) -> None:
    register_tool_plugin(SpeechToolPlugin, override=override)
