# © Artur Czarnecki. All rights reserved.

"""Optional Ultralytics YOLO vision adapter (Phase W-ML.3)."""

from __future__ import annotations

from intergrax.model_inference.adapters.opencv_vision import OpenCvVisionInferenceAdapter
from intergrax.model_inference.contracts import (
    ModelArtifact,
    VisionInferenceAdapter,
    VisionInferenceRequest,
    VisionInferenceResult,
)


class UltralyticsVisionInferenceAdapter(VisionInferenceAdapter):
    """
    YOLO adapter with OpenCV fallback when ``ultralytics`` is not installed.

    Production deployments should install ``ultralytics`` and provide a model path
    via artifact metadata key ``model_path``.
    """

    slug = "yolo_ultralytics"

    def __init__(self, *, fallback: VisionInferenceAdapter | None = None) -> None:
        self._fallback = fallback or OpenCvVisionInferenceAdapter()

    def detect(self, request: VisionInferenceRequest, *, artifact: ModelArtifact) -> VisionInferenceResult:
        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]
        except ImportError:
            return self._fallback.detect(request, artifact=artifact)

        model_path = artifact.metadata.get("model_path", "yolov8n.pt")
        model = YOLO(model_path)
        media_path = _resolve_media_path_from_request(request)
        results = model.predict(source=str(media_path), verbose=False)
        detections = _map_ultralytics_results(results, request_id=request.request_id)
        if not detections:
            return self._fallback.detect(request, artifact=artifact)
        return VisionInferenceResult(
            request_id=request.request_id,
            artifact_id=artifact.artifact_id,
            detections=detections[: request.top_k],
        )


def _resolve_media_path_from_request(request: VisionInferenceRequest) -> str:
    from intergrax.model_inference.adapters.opencv_vision import _resolve_media_path

    return str(_resolve_media_path(request.media_uri))


def _map_ultralytics_results(results: object, *, request_id: str) -> list:
    from intergrax.model_inference.contracts import VisionBoundingBox, VisionDetection

    mapped: list[VisionDetection] = []
    for result in results:  # type: ignore[union-attr]
        boxes = result.boxes  # type: ignore[attr-defined]
        if boxes is None:
            continue
        for box in boxes:
            xyxy = box.xyxy[0].tolist()  # type: ignore[index]
            conf = float(box.conf[0])  # type: ignore[index]
            cls_id = int(box.cls[0])  # type: ignore[index]
            name = result.names.get(cls_id, "object")  # type: ignore[attr-defined]
            mapped.append(
                VisionDetection(
                    label=str(name),
                    confidence=conf,
                    bbox=VisionBoundingBox(
                        x_min=float(xyxy[0]),
                        y_min=float(xyxy[1]),
                        x_max=float(xyxy[2]),
                        y_max=float(xyxy[3]),
                    ),
                )
            )
    return mapped
