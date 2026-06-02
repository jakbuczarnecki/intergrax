# © Artur Czarnecki. All rights reserved.

"""Declarative Tier-3 vision model selection (mirrors ``LLMProfile``)."""

from __future__ import annotations

import os
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.model_inference.adapters.huggingface_inference_vision import HuggingFaceInferenceVisionAdapter
from intergrax.model_inference.adapters.remote_serving import MlInferenceHostAdapter
from intergrax.model_inference.adapters.triton_vision import TritonVisionServingAdapter
from intergrax.model_inference.adapters.stub_ml import StubModelInferenceAdapter
from intergrax.model_inference.contracts import ModelArtifact, ModelArtifactFormat, VisionInferenceAdapter
from intergrax.model_inference.registry.core import ModelInferenceRegistry
from intergrax.model_inference.registry.vision_adapter_registry import VisionAdapterRegistry
from intergrax.model_inference.registry.vision_provider import VisionProvider

VISION_PROFILE_EXTRA_KEY = "vision_profile"

_DEFAULT_ARTIFACT_BY_PROVIDER: dict[VisionProvider, str] = {
    VisionProvider.STUB: "vision.stub.yolo",
    VisionProvider.OPENCV: "vision.opencv.onnx",
    VisionProvider.YOLO_ULTRALYTICS: "vision.yolo.ultralytics",
    VisionProvider.TRITON: "vision.triton.remote",
    VisionProvider.HUGGINGFACE_INFERENCE: "vision.huggingface.remote",
}

_DEFAULT_FORMAT_BY_PROVIDER: dict[VisionProvider, ModelArtifactFormat] = {
    VisionProvider.STUB: ModelArtifactFormat.ULTRALYTICS,
    VisionProvider.OPENCV: ModelArtifactFormat.ONNX,
    VisionProvider.YOLO_ULTRALYTICS: ModelArtifactFormat.ULTRALYTICS,
    VisionProvider.TRITON: ModelArtifactFormat.REMOTE,
    VisionProvider.HUGGINGFACE_INFERENCE: ModelArtifactFormat.REMOTE,
}


class VisionProfile(BaseModel):
    """
    Typed vision provider + artifact + constructor options for Tier-3 hosts.

    Example::

        profile = VisionProfile(provider=VisionProvider.OPENCV)
        adapter = profile.create_adapter()
        registry = profile.build_registry()
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    provider: VisionProvider
    artifact_id: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider", mode="before")
    @classmethod
    def _coerce_provider(cls, value: str | VisionProvider) -> VisionProvider:
        if isinstance(value, VisionProvider):
            return value
        if isinstance(value, str) and value.strip():
            raw = value.strip().lower()
            if raw == "opencv":
                return VisionProvider.OPENCV
            return VisionProvider(raw)
        raise ValueError("provider must be a non-empty VisionProvider or string slug")

    @property
    def resolved_artifact_id(self) -> str:
        if self.artifact_id:
            return self.artifact_id
        return _DEFAULT_ARTIFACT_BY_PROVIDER[self.provider]

    @property
    def adapter_slug(self) -> str:
        return self.create_adapter().slug

    def create_adapter(self, **overrides: Any) -> VisionInferenceAdapter:
        kwargs = {**self.options, **overrides}
        return VisionAdapterRegistry.create(self.provider, **kwargs)

    def build_artifact(self, *, adapter: VisionInferenceAdapter | None = None) -> ModelArtifact:
        resolved = adapter or self.create_adapter()
        return ModelArtifact(
            artifact_id=self.resolved_artifact_id,
            slug=resolved.slug,
            format=_DEFAULT_FORMAT_BY_PROVIDER[self.provider],
            metadata={"engine": self.provider.value, **{k: str(v) for k, v in self.options.items()}},
        )

    def build_registry(self, *, include_remote_placeholders: bool = True) -> ModelInferenceRegistry:
        """Materialize a registry with the configured primary vision adapter and ML stub."""
        registry = ModelInferenceRegistry()
        primary = self.create_adapter()
        registry.register_vision_adapter(primary)
        if include_remote_placeholders:
            registry.register_vision_adapter(TritonVisionServingAdapter())
            registry.register_vision_adapter(HuggingFaceInferenceVisionAdapter())
            registry.register_ml_adapter(MlInferenceHostAdapter())
        registry.register_ml_adapter(StubModelInferenceAdapter())
        registry.register_artifact(self.build_artifact(adapter=primary))
        registry.register_artifact(
            ModelArtifact(
                artifact_id="ml.stub.classifier",
                slug="sklearn_classifier",
                format=ModelArtifactFormat.SKLEARN,
                metadata={"engine": "stub"},
            )
        )
        return registry

    @classmethod
    def lab(cls) -> VisionProfile:
        """Laboratory default — OpenCV contour detector."""
        return cls(provider=VisionProvider.OPENCV)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> VisionProfile:
        return cls.model_validate(dict(data))


def vision_profile_from_env(*, prefix: str = "INTERGRAX_VISION") -> VisionProfile:
    """
    Build profile from environment:

    - ``{PREFIX}_PROVIDER`` or legacy ``INTERGRAX_VISION_ADAPTER`` (``stub`` | ``onnxruntime`` | ``yolo_ultralytics``)
    - ``{PREFIX}_ARTIFACT_ID`` (optional)
    """
    provider_raw = (
        os.getenv(f"{prefix}_PROVIDER")
        or os.getenv("INTERGRAX_VISION_ADAPTER")
        or VisionProvider.OPENCV.value
    ).strip()
    artifact_id = os.getenv(f"{prefix}_ARTIFACT_ID")
    return VisionProfile(
        provider=VisionProvider(provider_raw.lower()),
        artifact_id=artifact_id or None,
    )
