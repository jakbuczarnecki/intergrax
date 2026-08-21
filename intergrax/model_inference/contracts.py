# © Artur Czarnecki. All rights reserved.

"""Typed contracts for Plane C dedicated inference (Phase W-ML.3–W-ML.5)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class ModelArtifactFormat(str, Enum):
    ONNX = "onnx"
    SKLEARN = "sklearn"
    ULTRALYTICS = "ultralytics"
    REMOTE = "remote"


class ModelArtifact(BaseModel):
    artifact_id: str
    slug: str
    format: ModelArtifactFormat
    version: str = "1.0.0"
    input_schema_ref: str = ""
    output_schema_ref: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


class VisionBoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class VisionDetection(BaseModel):
    label: str
    confidence: float
    bbox: VisionBoundingBox


class MediaSourceKind(str, Enum):
    LOCAL_FILE = "local_file"


class MediaAuthorizationError(ValueError):
    """Raised when caller media is not authorized for inference access."""


class AuthorizedLocalMedia(BaseModel):
    """Resolved local media authorized through ToolWiringContext read roots."""

    source_kind: MediaSourceKind = MediaSourceKind.LOCAL_FILE
    resolved_path: Path
    remote_egress_permitted: bool = False


class VisionInferenceRequest(BaseModel):
    request_id: str
    artifact_id: str
    media_uri: str
    authorized_local_media: AuthorizedLocalMedia | None = None
    top_k: int = Field(default=5, ge=1, le=100)


class VisionInferenceResult(BaseModel):
    request_id: str
    artifact_id: str
    detections: list[VisionDetection] = Field(default_factory=list)


class VisionSegment(BaseModel):
    label: str
    confidence: float
    bbox: VisionBoundingBox


class VisionSegmentationResult(BaseModel):
    request_id: str
    artifact_id: str
    segments: list[VisionSegment] = Field(default_factory=list)


class VisionOcrRegion(BaseModel):
    text: str
    confidence: float
    bbox: VisionBoundingBox


class VisionOcrResult(BaseModel):
    request_id: str
    artifact_id: str
    regions: list[VisionOcrRegion] = Field(default_factory=list)


class InferenceRequest(BaseModel):
    request_id: str
    artifact_id: str
    features: dict[str, float] = Field(default_factory=dict)


class InferenceResult(BaseModel):
    request_id: str
    artifact_id: str
    predictions: dict[str, float] = Field(default_factory=dict)


class InferenceExplanationResult(BaseModel):
    request_id: str
    artifact_id: str
    predictions: dict[str, float] = Field(default_factory=dict)
    feature_importance: dict[str, float] = Field(default_factory=dict)


class VisionInferenceAdapter(ABC):
    """Plane C vision inference adapter contract."""

    slug: str

    @abstractmethod
    def detect(self, request: VisionInferenceRequest, *, artifact: ModelArtifact) -> VisionInferenceResult:
        raise NotImplementedError


class ExtendedVisionInferenceAdapter(VisionInferenceAdapter):
    """Vision adapters that also expose segmentation and OCR region APIs."""

    @abstractmethod
    def segment(self, request: VisionInferenceRequest, *, artifact: ModelArtifact) -> VisionSegmentationResult:
        raise NotImplementedError

    @abstractmethod
    def ocr_regions(self, request: VisionInferenceRequest, *, artifact: ModelArtifact) -> VisionOcrResult:
        raise NotImplementedError


class ModelInferenceAdapter(ABC):
    """Plane C classical ML inference adapter contract."""

    slug: str

    @abstractmethod
    def predict(self, request: InferenceRequest, *, artifact: ModelArtifact) -> InferenceResult:
        raise NotImplementedError

    def explain(self, request: InferenceRequest, *, artifact: ModelArtifact) -> InferenceExplanationResult:
        """Default explain path derives importance from normalized feature magnitudes."""
        prediction = self.predict(request, artifact=artifact)
        total = sum(abs(v) for v in request.features.values()) or 1.0
        importance = {key: abs(value) / total for key, value in request.features.items()}
        return InferenceExplanationResult(
            request_id=prediction.request_id,
            artifact_id=prediction.artifact_id,
            predictions=dict(prediction.predictions),
            feature_importance=importance,
        )
