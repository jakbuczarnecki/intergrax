# © Artur Czarnecki. All rights reserved.

"""OpenCV contour-based vision adapter for harness golden regression (Phase W-ML.3)."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2

from intergrax.model_inference.contracts import (
    ExtendedVisionInferenceAdapter,
    ModelArtifact,
    VisionBoundingBox,
    VisionDetection,
    VisionInferenceRequest,
    VisionInferenceResult,
    VisionOcrRegion,
    VisionOcrResult,
    VisionSegment,
    VisionSegmentationResult,
)


class OpenCvVisionInferenceAdapter(ExtendedVisionInferenceAdapter):
    """Deterministic object detection via contour analysis (no ONNX runtime required)."""

    slug = "onnxruntime"

    def detect(self, request: VisionInferenceRequest, *, artifact: ModelArtifact) -> VisionInferenceResult:
        regions = _contour_regions(request.media_uri, top_k=request.top_k)
        detections = [
            VisionDetection(label="contour.region", confidence=region.confidence, bbox=region.bbox)
            for region in regions
        ]
        return VisionInferenceResult(
            request_id=request.request_id,
            artifact_id=artifact.artifact_id,
            detections=detections,
        )

    def segment(self, request: VisionInferenceRequest, *, artifact: ModelArtifact) -> VisionSegmentationResult:
        regions = _contour_regions(request.media_uri, top_k=request.top_k)
        segments = [
            VisionSegment(label="contour.segment", confidence=region.confidence, bbox=region.bbox)
            for region in regions
        ]
        return VisionSegmentationResult(
            request_id=request.request_id,
            artifact_id=artifact.artifact_id,
            segments=segments,
        )

    def ocr_regions(self, request: VisionInferenceRequest, *, artifact: ModelArtifact) -> VisionOcrResult:
        image_path = _resolve_media_path(request.media_uri)
        text = _ocr_text(image_path)
        return VisionOcrResult(
            request_id=request.request_id,
            artifact_id=artifact.artifact_id,
            regions=[
                VisionOcrRegion(
                    text=text,
                    confidence=0.85,
                    bbox=VisionBoundingBox(x_min=0.0, y_min=0.0, x_max=1.0, y_max=0.25),
                )
            ],
        )


class _ContourRegion:
    __slots__ = ("confidence", "bbox")

    def __init__(self, *, confidence: float, bbox: VisionBoundingBox) -> None:
        self.confidence = confidence
        self.bbox = bbox


def _contour_regions(media_uri: str, *, top_k: int) -> list[_ContourRegion]:
    image_path = _resolve_media_path(media_uri)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    _, threshold = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image.shape[:2]
    regions: list[_ContourRegion] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:top_k]:
        area = cv2.contourArea(contour)
        if area < 10.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        regions.append(
            _ContourRegion(
                confidence=min(0.99, area / float(max(1, width * height))),
                bbox=VisionBoundingBox(
                    x_min=float(x) / float(width),
                    y_min=float(y) / float(height),
                    x_max=float(x + w) / float(width),
                    y_max=float(y + h) / float(height),
                ),
            )
        )
    return regions


def _ocr_text(image_path: Path) -> str:
    try:
        import pytesseract
    except ImportError:
        return "[opencv ocr stub]"
    image = cv2.imread(str(image_path))
    if image is None:
        return ""
    return (pytesseract.image_to_string(image) or "").strip() or "[empty]"


def _resolve_media_path(media_uri: str) -> Path:
    if media_uri.startswith("file://"):
        parsed = urlparse(media_uri)
        path_str = unquote(parsed.path)
        if path_str.startswith("/") and len(path_str) > 2 and path_str[2] == ":":
            path_str = path_str[1:]
        return Path(path_str)
    return Path(media_uri)
