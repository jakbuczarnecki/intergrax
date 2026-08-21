# © Artur Czarnecki. All rights reserved.

"""Triton / KServe HTTP vision adapter (Phase W-ML.4)."""

from __future__ import annotations

import base64
import os
from typing import Any

import httpx

from intergrax.model_inference.adapters.stub_vision import StubVisionInferenceAdapter
from intergrax.model_inference.contracts import (
    MediaAuthorizationError,
    ModelArtifact,
    VisionBoundingBox,
    VisionDetection,
    VisionInferenceAdapter,
    VisionInferenceRequest,
    VisionInferenceResult,
)
from intergrax.model_inference.media_boundary import require_remote_egress_bytes


class TritonVisionServingAdapter(VisionInferenceAdapter):
    """
    HTTP client for Triton inference server (v2 infer API subset).

    Falls back to stub detections when ``INTERGRAX_TRITON_URL`` is unset.
    """

    slug = "vision_serving"
    ENV_URL = "INTERGRAX_TRITON_URL"
    ENV_MODEL = "INTERGRAX_TRITON_MODEL"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout: float = 30.0,
        fallback: VisionInferenceAdapter | None = None,
    ) -> None:
        self._base_url = (base_url or os.getenv(self.ENV_URL) or "").strip().rstrip("/")
        self._model_name = (model_name or os.getenv(self.ENV_MODEL) or "yolo").strip()
        self._timeout = timeout
        self._fallback = fallback or StubVisionInferenceAdapter()

    def detect(self, request: VisionInferenceRequest, *, artifact: ModelArtifact) -> VisionInferenceResult:
        if not self._base_url:
            return self._fallback.detect(request, artifact=artifact)
        try:
            raw = require_remote_egress_bytes(request)
        except MediaAuthorizationError:
            raise
        encoded = base64.b64encode(raw).decode("ascii")
        url = f"{self._base_url}/v2/models/{self._model_name}/infer"
        payload: dict[str, Any] = {
            "inputs": [
                {
                    "name": "input",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [encoded],
                }
            ]
        }
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            body = response.json()
        detections = _parse_triton_detections(body, top_k=request.top_k)
        if not detections:
            return self._fallback.detect(request, artifact=artifact)
        return VisionInferenceResult(
            request_id=request.request_id,
            artifact_id=artifact.artifact_id,
            detections=detections,
        )


def _parse_triton_detections(body: dict[str, Any], *, top_k: int) -> list[VisionDetection]:
    outputs = body.get("outputs")
    if not isinstance(outputs, list):
        return []
    detections: list[VisionDetection] = []
    for output in outputs[:top_k]:
        if not isinstance(output, dict):
            continue
        label = str(output.get("label") or output.get("name") or "triton.detection")
        confidence = float(output.get("confidence", 0.9))
        bbox_raw = output.get("bbox")
        if isinstance(bbox_raw, dict):
            bbox = VisionBoundingBox(
                x_min=float(bbox_raw.get("x_min", 0.0)),
                y_min=float(bbox_raw.get("y_min", 0.0)),
                x_max=float(bbox_raw.get("x_max", 1.0)),
                y_max=float(bbox_raw.get("y_max", 1.0)),
            )
        else:
            bbox = VisionBoundingBox(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)
        detections.append(VisionDetection(label=label, confidence=confidence, bbox=bbox))
    return detections
