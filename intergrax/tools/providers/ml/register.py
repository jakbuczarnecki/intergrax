# © Artur Czarnecki. All rights reserved.

from intergrax.tools.providers.ml.bundle import ML_BUNDLE_ID, register_ml_tools
from intergrax.tools.registry.catalog import ToolBundleEntry, ToolBundleStatus, register_tool_bundle


def register_ml_tool_bundle(*, override: bool = False) -> None:
    register_tool_bundle(
        ToolBundleEntry(
            bundle_id=ML_BUNDLE_ID,
            tool_ids=("ml.predict", "ml.explain", "ml.batch_predict"),
            register=register_ml_tools,
            status=ToolBundleStatus.STABLE,
            description="Classical ML prediction tools.",
        ),
        override=override,
    )
