# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog registration for the RAG tool bundle."""

from __future__ import annotations

from intergrax.tools.registry.plugin_register import register_tool_plugin
from intergrax.tools.registry.shipped_plugins import RagToolPlugin


def register_rag_tool_bundle(*, override: bool = False) -> None:
    register_tool_plugin(RagToolPlugin, override=override)
