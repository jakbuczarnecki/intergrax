# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.tools.providers.braintrust.bundle import BRAINTRUST_BUNDLE_ID, register_braintrust_tools
from intergrax.tools.registry.catalog import ToolBundleEntry, ToolBundleStatus, register_tool_bundle


def register_braintrust_tool_bundle(*, override: bool = False) -> None:
    register_tool_bundle(
        ToolBundleEntry(
            bundle_id=BRAINTRUST_BUNDLE_ID,
            tool_ids=("braintrust.log_eval",),
            register=register_braintrust_tools,
            status=ToolBundleStatus.BETA,
            description="Braintrust eval logging tools.",
        ),
        override=override,
    )
