# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.tools.providers.confluence.bundle import CONFLUENCE_BUNDLE_ID, register_confluence_tools
from intergrax.tools.registry.catalog import ToolBundleEntry, ToolBundleStatus, register_tool_bundle


def register_confluence_tool_bundle(*, override: bool = False) -> None:
    register_tool_bundle(
        ToolBundleEntry(
            bundle_id=CONFLUENCE_BUNDLE_ID,
            tool_ids=(
                "confluence.get_page",
                "confluence.search_pages",
            ),
            register=register_confluence_tools,
            status=ToolBundleStatus.STABLE,
            description="Confluence wiki / knowledge tools.",
        ),
        override=override,
    )
