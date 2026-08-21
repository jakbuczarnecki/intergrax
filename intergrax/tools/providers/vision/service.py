# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import time
import uuid

from intergrax.model_inference.contracts import VisionInferenceRequest
from intergrax.runtime.observability.modality_counters import record_inference_ms, record_media_bytes
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
    measure_authorized_media_bytes,
    prepare_authorized_vision_media,
    resolve_executor,
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
    adapter = registry.get_vision_adapter(payload.adapter_slug)
    authorized = prepare_authorized_vision_media(
        ctx,
        profile,
        payload.media_uri,
        adapter_slug=payload.adapter_slug,
    )
    record_media_bytes(ctx, measure_authorized_media_bytes(authorized))
    executor = resolve_executor(ctx)
    artifact = registry.get_artifact(payload.artifact_id)
    request_id = uuid.uuid4().hex
    started = time.perf_counter()
    result = executor.run_detect(
        registry=registry,
        adapter=adapter,
        artifact=artifact,
        request=VisionInferenceRequest(
            request_id=request_id,
            artifact_id=artifact.artifact_id,
            media_uri=payload.media_uri,
            authorized_local_media=authorized,
            top_k=payload.top_k,
        ),
    )
    record_inference_ms(ctx, int((time.perf_counter() - started) * 1000), vision_invocation=True)
    return VisionDetectOutput(request_id=result.request_id, detections=result.detections)


def vision_segment(ctx: ToolWiringContext, payload: VisionSegmentInput) -> VisionSegmentOutput:
    profile = resolve_modality_profile(ctx)
    assert_artifact_allowed(profile, payload.artifact_id)
    registry = resolve_registry(ctx)
    adapter = as_extended_adapter(registry.get_vision_adapter(payload.adapter_slug))
    authorized = prepare_authorized_vision_media(
        ctx,
        profile,
        payload.media_uri,
        adapter_slug=payload.adapter_slug,
    )
    record_media_bytes(ctx, measure_authorized_media_bytes(authorized))
    executor = resolve_executor(ctx)
    artifact = registry.get_artifact(payload.artifact_id)
    request_id = uuid.uuid4().hex
    started = time.perf_counter()
    result = executor.run_segment(
        registry=registry,
        adapter=adapter,
        artifact=artifact,
        request=VisionInferenceRequest(
            request_id=request_id,
            artifact_id=artifact.artifact_id,
            media_uri=payload.media_uri,
            authorized_local_media=authorized,
            top_k=payload.top_k,
        ),
    )
    record_inference_ms(ctx, int((time.perf_counter() - started) * 1000), vision_invocation=True)
    return VisionSegmentOutput(request_id=result.request_id, segments=result.segments)


def vision_ocr_regions(ctx: ToolWiringContext, payload: VisionOcrRegionsInput) -> VisionOcrRegionsOutput:
    profile = resolve_modality_profile(ctx)
    assert_artifact_allowed(profile, payload.artifact_id)
    registry = resolve_registry(ctx)
    adapter = as_extended_adapter(registry.get_vision_adapter(payload.adapter_slug))
    authorized = prepare_authorized_vision_media(
        ctx,
        profile,
        payload.media_uri,
        adapter_slug=payload.adapter_slug,
    )
    record_media_bytes(ctx, measure_authorized_media_bytes(authorized))
    executor = resolve_executor(ctx)
    artifact = registry.get_artifact(payload.artifact_id)
    request_id = uuid.uuid4().hex
    started = time.perf_counter()
    result = executor.run_ocr_regions(
        registry=registry,
        adapter=adapter,
        artifact=artifact,
        request=VisionInferenceRequest(
            request_id=request_id,
            artifact_id=artifact.artifact_id,
            media_uri=payload.media_uri,
            authorized_local_media=authorized,
            top_k=payload.top_k,
        ),
    )
    record_inference_ms(ctx, int((time.perf_counter() - started) * 1000), vision_invocation=True)
    return VisionOcrRegionsOutput(request_id=result.request_id, regions=result.regions)
