# © Artur Czarnecki. All rights reserved.

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.vision.contracts import (
    VisionDetectInput,
    VisionDetectOutput,
    VisionOcrRegionsInput,
    VisionOcrRegionsOutput,
    VisionSegmentInput,
    VisionSegmentOutput,
)
from intergrax.tools.providers.vision.service import vision_detect, vision_ocr_regions, vision_segment


class VisionDetectHandler(ServiceToolHandler[VisionDetectInput, VisionDetectOutput]):
    _service = vision_detect


class VisionSegmentHandler(ServiceToolHandler[VisionSegmentInput, VisionSegmentOutput]):
    _service = vision_segment


class VisionOcrRegionsHandler(ServiceToolHandler[VisionOcrRegionsInput, VisionOcrRegionsOutput]):
    _service = vision_ocr_regions
