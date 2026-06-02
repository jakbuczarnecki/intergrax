# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.model_inference.contracts import VisionDetection


class VisionDetectInput(BaseModel):
    media_uri: str
    artifact_id: str = "vision.stub.yolo"
    adapter_slug: str = "yolo_ultralytics"
    top_k: int = Field(default=5, ge=1, le=100)


class VisionDetectOutput(BaseModel):
    request_id: str
    detections: list[VisionDetection] = Field(default_factory=list)
