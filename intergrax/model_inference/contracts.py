# © Artur Czarnecki. All rights reserved.

"""Typed contracts for Plane C dedicated inference (Phase W-ML.3–W-ML.5)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

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


class VisionInferenceRequest(BaseModel):
    request_id: str
    artifact_id: str
    media_uri: str
    top_k: int = Field(default=5, ge=1, le=100)


class VisionInferenceResult(BaseModel):
    request_id: str
    artifact_id: str
    detections: list[VisionDetection] = Field(default_factory=list)


class InferenceRequest(BaseModel):
    request_id: str
    artifact_id: str
    features: dict[str, float] = Field(default_factory=dict)


class InferenceResult(BaseModel):
    request_id: str
    artifact_id: str
    predictions: dict[str, float] = Field(default_factory=dict)


class VisionInferenceAdapter(ABC):
    """Plane C vision inference adapter contract."""

    slug: str

    @abstractmethod
    def detect(self, request: VisionInferenceRequest, *, artifact: ModelArtifact) -> VisionInferenceResult:
        raise NotImplementedError


class ModelInferenceAdapter(ABC):
    """Plane C classical ML inference adapter contract."""

    slug: str

    @abstractmethod
    def predict(self, request: InferenceRequest, *, artifact: ModelArtifact) -> InferenceResult:
        raise NotImplementedError
