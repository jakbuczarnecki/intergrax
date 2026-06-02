from __future__ import annotations

from intergrax.runtime.observability.modality_counters import (
    MODALITY_INVOCATION_COUNTERS_KEY,
    ModalityInvocationCounters,
    read_modality_counters,
    record_media_bytes,
    record_tts_characters,
)
from intergrax.runtime.observability.modality_tool_trace import modality_metrics_from_extras
from intergrax.tools.registry.wiring import ToolWiringContext


def test_record_media_and_tts_updates_typed_counters() -> None:
    ctx = ToolWiringContext()
    record_media_bytes(ctx, 2048)
    record_tts_characters(ctx, 120)
    counters = read_modality_counters(ctx)
    assert counters.media_bytes == 2048
    assert counters.tts_characters == 120
    metrics = modality_metrics_from_extras(ctx)
    assert metrics is not None
    assert metrics.media_bytes == 2048
    assert metrics.tts_characters == 120


def test_counters_stored_as_pydantic_model() -> None:
    ctx = ToolWiringContext()
    read_modality_counters(ctx)
    stored = ctx.extras[MODALITY_INVOCATION_COUNTERS_KEY]
    assert isinstance(stored, ModalityInvocationCounters)
