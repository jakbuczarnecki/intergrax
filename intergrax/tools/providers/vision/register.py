# © Artur Czarnecki. All rights reserved.

from intergrax.tools.providers.vision.bundle import VISION_BUNDLE_ID, register_vision_tools
from intergrax.tools.registry.catalog import ToolBundleEntry, ToolBundleStatus, register_tool_bundle


def register_vision_tool_bundle(*, override: bool = False) -> None:
    register_tool_bundle(
        ToolBundleEntry(
            bundle_id=VISION_BUNDLE_ID,
            tool_ids=("vision.detect",),
            register=register_vision_tools,
            status=ToolBundleStatus.STABLE,
            description="Dedicated vision inference tools.",
        ),
        override=override,
    )
