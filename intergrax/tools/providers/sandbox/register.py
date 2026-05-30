# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.tools.providers.sandbox.bundle import SANDBOX_BUNDLE_ID, register_sandbox_tools
from intergrax.tools.registry.catalog import ToolBundleEntry, ToolBundleStatus, register_tool_bundle


def register_sandbox_tool_bundle(*, override: bool = False) -> None:
    register_tool_bundle(
        ToolBundleEntry(
            bundle_id=SANDBOX_BUNDLE_ID,
            tool_ids=("sandbox.exec",),
            register=register_sandbox_tools,
            status=ToolBundleStatus.STABLE,
            description="Runtime sandbox execution tools.",
        ),
        override=override,
    )
