# © Artur Czarnecki. All rights reserved.

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.vision.contracts import VisionDetectInput, VisionDetectOutput
from intergrax.tools.providers.vision.handlers import VisionDetectHandler
from intergrax.tools.providers.vision.service import VISION_DETECT_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

VISION_BUNDLE_ID = "vision"


def register_vision_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=VISION_DETECT_TOOL_ID,
            name=VISION_DETECT_TOOL_ID,
            description="Run dedicated vision detection inference (Plane C).",
            description_short="Vision object detection.",
            input_schema=VisionDetectInput,
            output_schema=VisionDetectOutput,
            error_mapping={},
            side_effects=False,
            category="vision",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("vision", "cv", "modality"),
        ),
        VisionDetectHandler(ctx),
    )
