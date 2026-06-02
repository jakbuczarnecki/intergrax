# © Artur Czarnecki. All rights reserved.

"""Bridge modality counters from tool wiring context into trace diagnostics."""

from __future__ import annotations

from intergrax.runtime.observability.modality_counters import (
    MODALITY_INVOCATION_COUNTERS_KEY,
    ModalityInvocationCounters,
    read_modality_counters,
)
from intergrax.runtime.observability.modality_metrics import ModalityMetricsPayload
from intergrax.tools.core.handler import WiringContextToolHandler
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext


def modality_metrics_from_counters(counters: ModalityInvocationCounters) -> ModalityMetricsPayload:
    return ModalityMetricsPayload(
        inference_ms=counters.inference_ms,
        media_bytes=counters.media_bytes,
        tts_characters=counters.tts_characters,
        vision_detections=counters.vision_detections,
        ml_predictions=counters.ml_predictions,
    )


def modality_metrics_from_extras(ctx: ToolWiringContext) -> ModalityMetricsPayload | None:
    raw = ctx.extras.get(MODALITY_INVOCATION_COUNTERS_KEY)
    if raw is None:
        return None
    if isinstance(raw, ModalityInvocationCounters):
        counters = raw
    elif isinstance(raw, dict):
        counters = ModalityInvocationCounters.model_validate(raw)
    else:
        return None
    metrics = modality_metrics_from_counters(counters)
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
    return metrics


def modality_metrics_from_handler(handler: object) -> ModalityMetricsPayload | None:
    if isinstance(handler, WiringContextToolHandler):
        return modality_metrics_from_extras(handler._ctx)
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
    return metrics.model_dump()
