# © Artur Czarnecki. All rights reserved.

"""Typed per-invocation modality counters stored on ``ToolWiringContext.extras``."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.tools.registry.wiring import ToolWiringContext

MODALITY_INVOCATION_COUNTERS_KEY = "modality_invocation_counters"


class ModalityInvocationCounters(BaseModel):
    inference_ms: int = Field(default=0, ge=0)
    media_bytes: int = Field(default=0, ge=0)
    tts_characters: int = Field(default=0, ge=0)
    vision_detections: int = Field(default=0, ge=0)
    ml_predictions: int = Field(default=0, ge=0)


def read_modality_counters(ctx: ToolWiringContext) -> ModalityInvocationCounters:
    raw = ctx.extras.get(MODALITY_INVOCATION_COUNTERS_KEY)
    if isinstance(raw, ModalityInvocationCounters):
        return raw
    if isinstance(raw, dict):
        counters = ModalityInvocationCounters.model_validate(raw)
        ctx.extras[MODALITY_INVOCATION_COUNTERS_KEY] = counters
        return counters
    counters = ModalityInvocationCounters()
    ctx.extras[MODALITY_INVOCATION_COUNTERS_KEY] = counters
    return counters


def record_inference_ms(ctx: ToolWiringContext, elapsed_ms: int, *, vision_invocation: bool = False) -> None:
    counters = read_modality_counters(ctx)
    counters.inference_ms += max(0, elapsed_ms)
    if vision_invocation:
        counters.vision_detections += 1


def record_media_bytes(ctx: ToolWiringContext, byte_count: int) -> None:
    if byte_count <= 0:
        return
    counters = read_modality_counters(ctx)
    counters.media_bytes += byte_count


def record_tts_characters(ctx: ToolWiringContext, character_count: int) -> None:
    if character_count <= 0:
        return
    counters = read_modality_counters(ctx)
    counters.tts_characters += character_count


def record_ml_predictions(ctx: ToolWiringContext, prediction_count: int) -> None:
    if prediction_count <= 0:
        return
    counters = read_modality_counters(ctx)
    counters.ml_predictions += prediction_count
