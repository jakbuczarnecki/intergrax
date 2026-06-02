# © Artur Czarnecki. All rights reserved.

"""OpenCV contour-based vision adapter for harness golden regression (Phase W-ML.3)."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

from intergrax.model_inference.contracts import (
    ModelArtifact,
    VisionBoundingBox,
    VisionDetection,
    VisionInferenceAdapter,
    VisionInferenceRequest,
    VisionInferenceResult,
)


class OpenCvVisionInferenceAdapter(VisionInferenceAdapter):
    """Deterministic object detection via contour analysis (no ONNX runtime required)."""

    slug = "onnxruntime"

    def detect(self, request: VisionInferenceRequest, *, artifact: ModelArtifact) -> VisionInferenceResult:
        image_path = _resolve_media_path(request.media_uri)
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        _, threshold = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = image.shape[:2]
        detections: list[VisionDetection] = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[: request.top_k]:
            area = cv2.contourArea(contour)
            if area < 10.0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            detections.append(
                VisionDetection(
                    label="contour.region",
                    confidence=min(0.99, area / float(max(1, width * height))),
                    bbox=VisionBoundingBox(
                        x_min=float(x) / float(width),
                        y_min=float(y) / float(height),
                        x_max=float(x + w) / float(width),
                        y_max=float(y + h) / float(height),
                    ),
                )
            )
        return VisionInferenceResult(
            request_id=request.request_id,
            artifact_id=artifact.artifact_id,
            detections=detections,
        )


def _resolve_media_path(media_uri: str) -> Path:
    if media_uri.startswith("file://"):
        parsed = urlparse(media_uri)
        path_str = unquote(parsed.path)
        if path_str.startswith("/") and len(path_str) > 2 and path_str[2] == ":":
            path_str = path_str[1:]
        return Path(path_str)
    return Path(media_uri)
