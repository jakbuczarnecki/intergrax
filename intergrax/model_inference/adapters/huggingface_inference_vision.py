# © Artur Czarnecki. All rights reserved.

"""Hugging Face Inference API vision adapter (Phase W-ML.4)."""

from __future__ import annotations

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


class HuggingFaceInferenceVisionAdapter(VisionInferenceAdapter):
    """Object-detection via Hugging Face Inference Providers API."""

    slug = "huggingface_inference"
    ENV_API_KEY = "HUGGINGFACE_API_KEY"
    ENV_MODEL = "INTERGRAX_HF_VISION_MODEL"
    DEFAULT_MODEL = "facebook/detr-resnet-50"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_id: str | None = None,
        timeout: float = 60.0,
        fallback: VisionInferenceAdapter | None = None,
    ) -> None:
        resolved_key = (api_key or os.getenv(self.ENV_API_KEY) or "").strip()
        self._api_key = resolved_key
        self._model_id = (model_id or os.getenv(self.ENV_MODEL) or self.DEFAULT_MODEL).strip()
        self._timeout = timeout
        self._fallback = fallback or StubVisionInferenceAdapter()

    def detect(self, request: VisionInferenceRequest, *, artifact: ModelArtifact) -> VisionInferenceResult:
        if not self._api_key:
            return self._fallback.detect(request, artifact=artifact)
        try:
            raw = require_remote_egress_bytes(request)
        except MediaAuthorizationError:
            raise
        url = f"https://api-inference.huggingface.co/models/{self._model_id}"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(url, headers=headers, content=raw)
            response.raise_for_status()
            body = response.json()
        detections = _parse_hf_detections(body, top_k=request.top_k)
        if not detections:
            return self._fallback.detect(request, artifact=artifact)
        return VisionInferenceResult(
            request_id=request.request_id,
            artifact_id=artifact.artifact_id,
            detections=detections,
        )


def _parse_hf_detections(body: Any, *, top_k: int) -> list[VisionDetection]:
    rows: list[Any]
    if isinstance(body, list):
        rows = body
    elif isinstance(body, dict) and isinstance(body.get("predictions"), list):
        rows = body["predictions"]
    else:
        return []
    detections: list[VisionDetection] = []
    for row in rows[:top_k]:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label") or row.get("class") or "hf.detection")
        confidence = float(row.get("score", row.get("confidence", 0.9)))
        box = row.get("box") if isinstance(row.get("box"), dict) else row
        if isinstance(box, dict):
            bbox = VisionBoundingBox(
                x_min=float(box.get("xmin", box.get("x_min", 0.0))) / 1000.0,
                y_min=float(box.get("ymin", box.get("y_min", 0.0))) / 1000.0,
                x_max=float(box.get("xmax", box.get("x_max", 1.0))) / 1000.0,
                y_max=float(box.get("ymax", box.get("y_max", 1.0))) / 1000.0,
            )
        else:
            bbox = VisionBoundingBox(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)
        detections.append(VisionDetection(label=label, confidence=confidence, bbox=bbox))
    return detections
