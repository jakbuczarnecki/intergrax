# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.tools.providers.notify.bundle import NOTIFY_BUNDLE_ID, register_notify_tools
from intergrax.tools.registry.catalog import ToolBundleEntry, ToolBundleStatus, register_tool_bundle


def register_notify_tool_bundle(*, override: bool = False) -> None:
    register_tool_bundle(
        ToolBundleEntry(
            bundle_id=NOTIFY_BUNDLE_ID,
            tool_ids=("notify.send",),
            register=register_notify_tools,
            status=ToolBundleStatus.STABLE,
            description="Outbound notification tools.",
        ),
        override=override,
    )
