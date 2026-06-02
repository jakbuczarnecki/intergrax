# © Artur Czarnecki. All rights reserved.

"""Bridge modality counters from tool wiring context into trace diagnostics."""

from __future__ import annotations

from typing import Mapping

from intergrax.runtime.observability.modality_metrics import ModalityMetricsPayload
from intergrax.tools.core.handler import WiringContextToolHandler
from intergrax.tools.registry import ToolRegistry

MODALITY_INVOCATION_COUNTERS_KEY = "modality_invocation_counters"


def modality_metrics_from_extras(extras: Mapping[str, object]) -> ModalityMetricsPayload | None:
    raw = extras.get(MODALITY_INVOCATION_COUNTERS_KEY)
    if not isinstance(raw, dict):
        return None
    return ModalityMetricsPayload(
        inference_ms=int(raw.get("inference_ms", 0) or 0),
        media_bytes=int(raw.get("media_bytes", 0) or 0),
        tts_characters=int(raw.get("tts_characters", 0) or 0),
        vision_detections=int(raw.get("vision_detections", 0) or 0),
        ml_predictions=int(raw.get("ml_predictions", 0) or 0),
    )


def modality_metrics_from_handler(handler: object) -> ModalityMetricsPayload | None:
    if isinstance(handler, WiringContextToolHandler):
        return modality_metrics_from_extras(handler._ctx.extras)
    return None


def consume_modality_metrics_for_tool(registry: ToolRegistry, tool_id: str) -> ModalityMetricsPayload | None:
    """Read and clear per-invocation modality counters for a catalog tool."""
    registered = registry.get(tool_id)
    metrics = modality_metrics_from_handler(registered.handler)
    if isinstance(registered.handler, WiringContextToolHandler):
        registered.handler._ctx.extras.pop(MODALITY_INVOCATION_COUNTERS_KEY, None)
    return metrics


def modality_metrics_dict(metrics: ModalityMetricsPayload | None) -> dict[str, int] | None:
    if metrics is None:
        return None
    if not any(
        (
            metrics.inference_ms,
            metrics.media_bytes,
            metrics.tts_characters,
            metrics.vision_detections,
            metrics.ml_predictions,
        )
    ):
        return None
    return metrics.model_dump()
