# © Artur Czarnecki. All rights reserved.

"""Modality metrics payload for TASK_COMPLETED export (Phase W-ML.7)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.runtime.events.runtime_event import RuntimeEvent


class ModalityMetricsPayload(BaseModel):
    inference_ms: int = 0
    media_bytes: int = 0
    tts_characters: int = 0
    vision_detections: int = 0
    ml_predictions: int = 0


def extract_modality_metrics(event: RuntimeEvent) -> ModalityMetricsPayload:
    """Extract modality counters from structured runtime event payload."""
    payload = event.payload if isinstance(event.payload, dict) else {}
    modality_raw = payload.get("modality_metrics")
    if isinstance(modality_raw, dict):
        return ModalityMetricsPayload.model_validate(modality_raw)
    return ModalityMetricsPayload(
        inference_ms=int(payload.get("inference_ms", 0) or 0),
        media_bytes=int(payload.get("media_bytes", 0) or 0),
        tts_characters=int(payload.get("tts_characters", 0) or 0),
        vision_detections=int(payload.get("vision_detections", 0) or 0),
        ml_predictions=int(payload.get("ml_predictions", 0) or 0),
    )
