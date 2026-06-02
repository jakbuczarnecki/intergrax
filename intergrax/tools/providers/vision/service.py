# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import uuid

from intergrax.model_inference import build_default_model_inference_registry
from intergrax.model_inference.contracts import VisionInferenceRequest
from intergrax.tools.providers.vision.contracts import VisionDetectInput, VisionDetectOutput

VISION_DETECT_TOOL_ID = "vision.detect"


def vision_detect(payload: VisionDetectInput) -> VisionDetectOutput:
    registry = build_default_model_inference_registry()
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
