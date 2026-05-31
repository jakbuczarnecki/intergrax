# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.tools.providers.jira.bundle import JIRA_BUNDLE_ID, JIRA_TOOL_IDS, register_jira_tools
from intergrax.tools.registry.catalog import ToolBundleEntry, ToolBundleStatus, register_tool_bundle


def register_jira_tool_bundle(*, override: bool = False) -> None:
    register_tool_bundle(
        ToolBundleEntry(
            bundle_id=JIRA_BUNDLE_ID,
            tool_ids=JIRA_TOOL_IDS,
            register=register_jira_tools,
            status=ToolBundleStatus.STABLE,
            description="Jira issue tracker tools.",
        ),
        override=override,
    )
