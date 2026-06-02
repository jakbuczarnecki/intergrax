# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import uuid

from intergrax.model_inference.contracts import VisionInferenceRequest
from intergrax.model_inference.registry import ModelInferenceRegistry
from intergrax.tools.providers.speech.backends import MODEL_INFERENCE_REGISTRY_EXTRA_KEY
from intergrax.tools.providers.vision.contracts import VisionDetectInput, VisionDetectOutput
from intergrax.tools.registry.wiring import ToolWiringContext

VISION_DETECT_TOOL_ID = "vision.detect"


def _resolve_registry(ctx: ToolWiringContext) -> ModelInferenceRegistry:
    registry = ctx.extras.get(MODEL_INFERENCE_REGISTRY_EXTRA_KEY)
    if registry is None:
        from intergrax.model_inference.bootstrap import build_harness_model_inference_registry

        return build_harness_model_inference_registry()
    return registry


def vision_detect(ctx: ToolWiringContext, payload: VisionDetectInput) -> VisionDetectOutput:
    registry = _resolve_registry(ctx)
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
