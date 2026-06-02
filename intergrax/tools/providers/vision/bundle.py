# © Artur Czarnecki. All rights reserved.

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.vision.contracts import (
    VisionDetectInput,
    VisionDetectOutput,
    VisionOcrRegionsInput,
    VisionOcrRegionsOutput,
    VisionSegmentInput,
    VisionSegmentOutput,
)
from intergrax.tools.providers.vision.handlers import (
    VisionDetectHandler,
    VisionOcrRegionsHandler,
    VisionSegmentHandler,
)
from intergrax.tools.providers.vision.service import (
    VISION_DETECT_TOOL_ID,
    VISION_OCR_REGIONS_TOOL_ID,
    VISION_SEGMENT_TOOL_ID,
)
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
    registry.register(
        ToolContract(
            tool_id=VISION_SEGMENT_TOOL_ID,
            name=VISION_SEGMENT_TOOL_ID,
            description="Run dedicated vision segmentation inference (Plane C).",
            description_short="Vision segmentation.",
            input_schema=VisionSegmentInput,
            output_schema=VisionSegmentOutput,
            error_mapping={},
            side_effects=False,
            category="vision",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("vision", "cv", "modality"),
        ),
        VisionSegmentHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=VISION_OCR_REGIONS_TOOL_ID,
            name=VISION_OCR_REGIONS_TOOL_ID,
            description="Extract OCR text regions from media (Plane C).",
            description_short="Vision OCR regions.",
            input_schema=VisionOcrRegionsInput,
            output_schema=VisionOcrRegionsOutput,
            error_mapping={},
            side_effects=False,
            category="vision",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("vision", "ocr", "modality"),
        ),
        VisionOcrRegionsHandler(ctx),
    )
