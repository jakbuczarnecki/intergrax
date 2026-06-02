# © Artur Czarnecki. All rights reserved.

"""Shared vision tool wiring helpers."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

from intergrax.model_inference.contracts import ExtendedVisionInferenceAdapter, VisionInferenceAdapter
from intergrax.model_inference.execution import (
    MODALITY_EXECUTOR_EXTRA_KEY,
    ModalityInferenceExecutor,
    build_modality_inference_executor,
)
from intergrax.model_inference.registry import ModelInferenceRegistry
from intergrax.model_inference.adapters.stub_vision import StubVisionInferenceAdapter
from intergrax.runtime.modality.modality_profile import MODALITY_PROFILE_EXTRA_KEY, ModalityProfile
from intergrax.tools.providers.speech.backends import MODEL_INFERENCE_REGISTRY_EXTRA_KEY
from intergrax.tools.registry.wiring import ToolWiringContext


def resolve_registry(ctx: ToolWiringContext) -> ModelInferenceRegistry:
    registry = ctx.extras.get(MODEL_INFERENCE_REGISTRY_EXTRA_KEY)
    if registry is None:
        from intergrax.model_inference.bootstrap import build_harness_model_inference_registry

        return build_harness_model_inference_registry()
    return registry


def resolve_executor(ctx: ToolWiringContext) -> ModalityInferenceExecutor:
    executor = ctx.extras.get(MODALITY_EXECUTOR_EXTRA_KEY)
    if isinstance(executor, ModalityInferenceExecutor):
        return executor
    return build_modality_inference_executor()


def resolve_modality_profile(ctx: ToolWiringContext) -> ModalityProfile | None:
    raw = ctx.extras.get(MODALITY_PROFILE_EXTRA_KEY)
    if isinstance(raw, ModalityProfile):
        return raw
    return None


def assert_artifact_allowed(profile: ModalityProfile | None, artifact_id: str) -> None:
    if profile is None:
        return
    if profile.vision_model_ids and artifact_id not in profile.vision_model_ids:
        raise ValueError(f"artifact_id {artifact_id!r} not allowed by ModalityProfile {profile.profile_id!r}")


def measure_media_bytes(media_uri: str) -> int:
    """Return on-disk byte size for a local media URI, or ``0`` when unknown."""
    path = _resolve_media_path(media_uri)
    if not path.is_file():
        return 0
    return path.stat().st_size


def assert_media_within_limit(profile: ModalityProfile | None, media_uri: str) -> None:
    if profile is None or profile.max_media_bytes is None:
        return
    size = measure_media_bytes(media_uri)
    if size <= 0:
        return
    if size > profile.max_media_bytes:
        raise ValueError(
            f"media size {size} bytes exceeds ModalityProfile max_media_bytes={profile.max_media_bytes}"
        )


def as_extended_adapter(adapter: VisionInferenceAdapter) -> ExtendedVisionInferenceAdapter:
    if isinstance(adapter, ExtendedVisionInferenceAdapter):
        return adapter
    return StubVisionInferenceAdapter()


def _resolve_media_path(media_uri: str) -> Path:
    if media_uri.startswith("file://"):
        parsed = urlparse(media_uri)
        path_str = unquote(parsed.path)
        if path_str.startswith("/") and len(path_str) > 2 and path_str[2] == ":":
            path_str = path_str[1:]
        return Path(path_str)
    return Path(media_uri)
