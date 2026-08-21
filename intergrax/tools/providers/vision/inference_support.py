# © Artur Czarnecki. All rights reserved.

"""Shared vision tool wiring helpers."""

from __future__ import annotations

from intergrax.model_inference.contracts import AuthorizedLocalMedia, MediaAuthorizationError, ExtendedVisionInferenceAdapter, VisionInferenceAdapter
from intergrax.model_inference.execution import (
    MODALITY_EXECUTOR_EXTRA_KEY,
    ModalityInferenceExecutor,
    build_modality_inference_executor,
)
from intergrax.model_inference.media_boundary import (
    adapter_requires_remote_egress,
    parse_local_media_candidate,
    resolve_authorized_local_media,
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


def authorize_vision_media(
    ctx: ToolWiringContext,
    media_uri: str,
    *,
    adapter_slug: str,
) -> AuthorizedLocalMedia:
    remote_egress = adapter_requires_remote_egress(adapter_slug)
    return resolve_authorized_local_media(
        media_uri,
        roots=ctx.read_allowlist_roots,
        remote_egress=remote_egress,
    )


def measure_authorized_media_bytes(media: AuthorizedLocalMedia) -> int:
    path = media.resolved_path
    if not path.is_file():
        return 0
    return path.stat().st_size


def measure_media_bytes(media_uri: str) -> int:
    """Return byte size for local file URIs; ``0`` for non-local schemes (speech stub URIs)."""
    stripped = media_uri.strip()
    if not stripped or ("://" in stripped and not stripped.startswith("file://")):
        return 0
    try:
        candidate = parse_local_media_candidate(stripped)
        resolved = candidate.expanduser().resolve()
    except MediaAuthorizationError:
        return 0
    if not resolved.is_file():
        return 0
    return resolved.stat().st_size


def assert_media_within_limit(profile: ModalityProfile | None, media: AuthorizedLocalMedia) -> None:
    if profile is None or profile.max_media_bytes is None:
        return
    size = measure_authorized_media_bytes(media)
    if size <= 0:
        return
    if size > profile.max_media_bytes:
        raise ValueError(
            f"media size {size} bytes exceeds ModalityProfile max_media_bytes={profile.max_media_bytes}"
        )


def prepare_authorized_vision_media(
    ctx: ToolWiringContext,
    profile: ModalityProfile | None,
    media_uri: str,
    *,
    adapter_slug: str,
) -> AuthorizedLocalMedia:
    authorized = authorize_vision_media(ctx, media_uri, adapter_slug=adapter_slug)
    assert_media_within_limit(profile, authorized)
    return authorized


def as_extended_adapter(adapter: VisionInferenceAdapter) -> ExtendedVisionInferenceAdapter:
    if isinstance(adapter, ExtendedVisionInferenceAdapter):
        return adapter
    return StubVisionInferenceAdapter()


__all__ = [
    "MediaAuthorizationError",
    "assert_artifact_allowed",
    "assert_media_within_limit",
    "as_extended_adapter",
    "authorize_vision_media",
    "measure_authorized_media_bytes",
    "measure_media_bytes",
    "prepare_authorized_vision_media",
    "resolve_executor",
    "resolve_modality_profile",
    "resolve_registry",
]
