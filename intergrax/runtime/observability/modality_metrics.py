# © Artur Czarnecki. All rights reserved.

"""Modality metrics payload for TASK_COMPLETED export (Phase W-ML.7)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from pydantic import BaseModel, Field

from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.nexus.tracing.persistence_models import SerializedTraceEvent


class ModalityMetricsPayload(BaseModel):
    inference_ms: int = Field(default=0, ge=0)
    media_bytes: int = Field(default=0, ge=0)
    tts_characters: int = Field(default=0, ge=0)
    vision_detections: int = Field(default=0, ge=0)
    ml_predictions: int = Field(default=0, ge=0)

    def has_values(self) -> bool:
        return any(
            (
                self.inference_ms,
                self.media_bytes,
                self.tts_characters,
                self.vision_detections,
                self.ml_predictions,
            )
        )


def aggregate_modality_metrics_from_trace_events(
    events: Sequence[SerializedTraceEvent | Dict[str, Any]],
) -> ModalityMetricsPayload:
    """Sum modality counters from ``tool_invocation_end`` trace events (fallback: last payload)."""
    aggregated = ModalityMetricsPayload()
    found = False
    for event in events:
        payload = _trace_event_payload(event)
        step = event.step if isinstance(event, SerializedTraceEvent) else str(event.get("step", ""))
        if step != "tool_invocation_end":
            continue
        modality_raw = payload.get("modality_metrics")
        if isinstance(modality_raw, dict):
            partial = ModalityMetricsPayload.model_validate(modality_raw)
            aggregated = ModalityMetricsPayload(
                inference_ms=aggregated.inference_ms + partial.inference_ms,
                media_bytes=aggregated.media_bytes + partial.media_bytes,
                tts_characters=aggregated.tts_characters + partial.tts_characters,
                vision_detections=aggregated.vision_detections + partial.vision_detections,
                ml_predictions=aggregated.ml_predictions + partial.ml_predictions,
            )
            found = True
    if found:
        return aggregated
    for event in reversed(list(events)):
        payload = _trace_event_payload(event)
        if "modality_metrics" in payload:
            return ModalityMetricsPayload.model_validate(payload["modality_metrics"])
    return ModalityMetricsPayload()


def _trace_event_payload(event: SerializedTraceEvent | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(event, dict):
        raw = event.get("payload")
    else:
        raw = event.payload
    return raw if isinstance(raw, dict) else {}


def build_task_completed_modality_payload(
    events: Sequence[SerializedTraceEvent | Dict[str, Any]],
) -> Dict[str, Any] | None:
    """Runtime ``TASK_COMPLETED`` payload fragment when trace contains modality counters."""
    metrics = aggregate_modality_metrics_from_trace_events(events)
    if not metrics.has_values():
        return None
    return {"modality_metrics": metrics.model_dump()}


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
