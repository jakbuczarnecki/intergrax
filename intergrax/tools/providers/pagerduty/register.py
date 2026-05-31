# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.tools.providers.pagerduty.bundle import PAGERDUTY_BUNDLE_ID, register_pagerduty_tools
from intergrax.tools.registry.catalog import ToolBundleEntry, ToolBundleStatus, register_tool_bundle


def register_pagerduty_tool_bundle(*, override: bool = False) -> None:
    register_tool_bundle(
        ToolBundleEntry(
            bundle_id=PAGERDUTY_BUNDLE_ID,
            tool_ids=("pagerduty.trigger_incident",),
            register=register_pagerduty_tools,
            status=ToolBundleStatus.BETA,
            description="PagerDuty escalation tools.",
        ),
        override=override,
    )
