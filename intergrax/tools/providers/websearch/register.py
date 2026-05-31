# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog registration for the websearch tool bundle."""

from __future__ import annotations

from intergrax.tools.providers.websearch.bundle import WEBSEARCH_BUNDLE_ID, WEBSEARCH_TOOL_IDS, register_websearch_tools
from intergrax.tools.registry.catalog import ToolBundleEntry, ToolBundleStatus, register_tool_bundle


def register_websearch_tool_bundle(*, override: bool = False) -> None:
    register_tool_bundle(
        ToolBundleEntry(
            bundle_id=WEBSEARCH_BUNDLE_ID,
            tool_ids=WEBSEARCH_TOOL_IDS,
            register=register_websearch_tools,
            status=ToolBundleStatus.STABLE,
            description="Web research tools (live search APIs).",
        ),
        override=override,
    )
