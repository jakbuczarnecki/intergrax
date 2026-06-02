# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import uuid

from intergrax.model_inference.contracts import VisionInferenceRequest
from intergrax.tools.providers.vision.contracts import (
    VisionDetectInput,
    VisionDetectOutput,
    VisionOcrRegionsInput,
    VisionOcrRegionsOutput,
    VisionSegmentInput,
    VisionSegmentOutput,
)
from intergrax.tools.providers.vision.inference_support import (
    as_extended_adapter,
    assert_artifact_allowed,
    resolve_modality_profile,
    resolve_registry,
)
from intergrax.tools.registry.wiring import ToolWiringContext

VISION_DETECT_TOOL_ID = "vision.detect"
VISION_SEGMENT_TOOL_ID = "vision.segment"
VISION_OCR_REGIONS_TOOL_ID = "vision.ocr_regions"


def vision_detect(ctx: ToolWiringContext, payload: VisionDetectInput) -> VisionDetectOutput:
    profile = resolve_modality_profile(ctx)
    assert_artifact_allowed(profile, payload.artifact_id)
    registry = resolve_registry(ctx)
    artifact = registry.get_artifact(payload.artifact_id)
    adapter = registry.get_vision_adapter(payload.adapter_slug)
    request_id = uuid.uuid4().hex
    result = adapter.detect(
        VisionInferenceRequest(
            request_id=request_id,
            artifact_id=artifact.artifact_id,
            media_uri=payload.media_uri,
            top_k=payload.top_k,
        ),
        artifact=artifact,
    )
    return VisionDetectOutput(request_id=result.request_id, detections=result.detections)


def vision_segment(ctx: ToolWiringContext, payload: VisionSegmentInput) -> VisionSegmentOutput:
    profile = resolve_modality_profile(ctx)
    assert_artifact_allowed(profile, payload.artifact_id)
    registry = resolve_registry(ctx)
    artifact = registry.get_artifact(payload.artifact_id)
    adapter = as_extended_adapter(registry.get_vision_adapter(payload.adapter_slug))
    request_id = uuid.uuid4().hex
    result = adapter.segment(
        VisionInferenceRequest(
            request_id=request_id,
            artifact_id=artifact.artifact_id,
            media_uri=payload.media_uri,
            top_k=payload.top_k,
        ),
        artifact=artifact,
    )
    return VisionSegmentOutput(request_id=result.request_id, segments=result.segments)


def vision_ocr_regions(ctx: ToolWiringContext, payload: VisionOcrRegionsInput) -> VisionOcrRegionsOutput:
    profile = resolve_modality_profile(ctx)
    assert_artifact_allowed(profile, payload.artifact_id)
    registry = resolve_registry(ctx)
    artifact = registry.get_artifact(payload.artifact_id)
    adapter = as_extended_adapter(registry.get_vision_adapter(payload.adapter_slug))
    request_id = uuid.uuid4().hex
    result = adapter.ocr_regions(
        VisionInferenceRequest(
            request_id=request_id,
            artifact_id=artifact.artifact_id,
            media_uri=payload.media_uri,
            top_k=payload.top_k,
        ),
        artifact=artifact,
    )
    return VisionOcrRegionsOutput(request_id=result.request_id, regions=result.regions)
