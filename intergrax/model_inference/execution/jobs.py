# © Artur Czarnecki. All rights reserved.

"""Serializable modality inference jobs for distributed workers."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from intergrax.model_inference.contracts import (
    InferenceRequest,
    InferenceResult,
    VisionInferenceRequest,
    VisionInferenceResult,
    VisionOcrResult,
    VisionSegmentationResult,
)


class ModalityJobKind(str, Enum):
    DETECT = "detect"
    SEGMENT = "segment"
    OCR_REGIONS = "ocr_regions"
    PREDICT = "predict"


class ModalityDetectJob(BaseModel):
    kind: ModalityJobKind = ModalityJobKind.DETECT
    adapter_slug: str
    artifact_id: str
    request: VisionInferenceRequest


class ModalitySegmentJob(BaseModel):
    kind: ModalityJobKind = ModalityJobKind.SEGMENT
    adapter_slug: str
    artifact_id: str
    request: VisionInferenceRequest


class ModalityOcrJob(BaseModel):
    kind: ModalityJobKind = ModalityJobKind.OCR_REGIONS
    adapter_slug: str
    artifact_id: str
    request: VisionInferenceRequest


class ModalityPredictJob(BaseModel):
    kind: ModalityJobKind = ModalityJobKind.PREDICT
    adapter_slug: str
    artifact_id: str
    request: InferenceRequest


class ModalityJobResult(BaseModel):
    detect: VisionInferenceResult | None = None
    segment: VisionSegmentationResult | None = None
    ocr: VisionOcrResult | None = None
    predict: InferenceResult | None = None
    error: str | None = None
