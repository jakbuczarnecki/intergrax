# © Artur Czarnecki. All rights reserved.

"""Catalog registration for the OpenAI vector store tool bundle."""

from __future__ import annotations

from intergrax.tools.registry.plugin_register import register_tool_plugin
from intergrax.tools.registry.shipped_plugins import OpenaiVectorStoreToolPlugin


def register_openai_vector_store_tool_bundle(*, override: bool = False) -> None:
    register_tool_plugin(OpenaiVectorStoreToolPlugin, override=override)
