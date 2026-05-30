# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.tools.providers.observability.bundle import OBSERVABILITY_BUNDLE_ID, register_observability_tools
from intergrax.tools.registry.catalog import ToolBundleEntry, ToolBundleStatus, register_tool_bundle


def register_observability_tool_bundle(*, override: bool = False) -> None:
    register_tool_bundle(
        ToolBundleEntry(
            bundle_id=OBSERVABILITY_BUNDLE_ID,
            tool_ids=("metrics.query_instant", "logs.search"),
            register=register_observability_tools,
            status=ToolBundleStatus.BETA,
            description="Metrics and log search tools.",
        ),
        override=override,
    )
