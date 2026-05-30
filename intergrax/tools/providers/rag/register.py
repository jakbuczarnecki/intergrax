# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog registration for the RAG tool bundle."""

from __future__ import annotations

from intergrax.tools.providers.rag.bundle import RAG_BUNDLE_ID, RAG_TOOL_IDS, register_rag_tools
from intergrax.tools.registry.catalog import ToolBundleEntry, ToolBundleStatus, register_tool_bundle


def register_rag_tool_bundle(*, override: bool = False) -> None:
    register_tool_bundle(
        ToolBundleEntry(
            bundle_id=RAG_BUNDLE_ID,
            tool_ids=RAG_TOOL_IDS,
            register=register_rag_tools,
            status=ToolBundleStatus.STABLE,
            description="Vector retrieval tools for indexed documents (RAG).",
        ),
        override=override,
    )
