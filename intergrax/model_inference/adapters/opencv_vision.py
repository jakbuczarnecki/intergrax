# © Artur Czarnecki. All rights reserved.

"""OpenCV contour-based vision adapter for harness golden regression (Phase W-ML.3)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationDependencyError
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
from intergrax.model_inference.media_boundary import local_media_path_from_request


def _import_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        if exc.name == "cv2":
            raise IntegrationDependencyError(
                "OpenCV vision requires optional dependency 'opencv-python-headless'. "
                "Install Intergrax-ai[media-video].",
                integration_name="opencv",
            ) from exc
        raise
    return cv2


class OpenCvVisionInferenceAdapter(ExtendedVisionInferenceAdapter):
    """Deterministic object detection via contour analysis (no ONNX runtime required)."""

    slug = "onnxruntime"

    def detect(self, request: VisionInferenceRequest, *, artifact: ModelArtifact) -> VisionInferenceResult:
        regions = _contour_regions(request, top_k=request.top_k)
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
        regions = _contour_regions(request, top_k=request.top_k)
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
        image_path = local_media_path_from_request(request)
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


def _contour_regions(request: VisionInferenceRequest, *, top_k: int) -> list[_ContourRegion]:
    cv2 = _import_cv2()
    image_path = local_media_path_from_request(request)
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


def _ocr_text(image_path) -> str:
    try:
        import pytesseract
    except ImportError:
        return "[opencv ocr stub]"
    image = _import_cv2().imread(str(image_path))
    if image is None:
        return ""
    return (pytesseract.image_to_string(image) or "").strip() or "[empty]"
