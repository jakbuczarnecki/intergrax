# © Artur Czarnecki. All rights reserved.

"""Factory registry for ``VisionInferenceAdapter`` implementations."""

from __future__ import annotations

import importlib
from typing import Any, Callable

from intergrax.model_inference.contracts import VisionInferenceAdapter
from intergrax.model_inference.registry.vision_provider import VisionProvider

_BUILTIN_VISION_ADAPTERS: dict[str, tuple[str, str]] = {
    VisionProvider.STUB.value: (
        "intergrax.model_inference.adapters.stub_vision",
        "StubVisionInferenceAdapter",
    ),
    VisionProvider.OPENCV.value: (
        "intergrax.model_inference.adapters.opencv_vision",
        "OpenCvVisionInferenceAdapter",
    ),
    VisionProvider.YOLO_ULTRALYTICS.value: (
        "intergrax.model_inference.adapters.ultralytics_vision",
        "UltralyticsVisionInferenceAdapter",
    ),
    VisionProvider.TRITON.value: (
        "intergrax.model_inference.adapters.triton_vision",
        "TritonVisionServingAdapter",
    ),
    VisionProvider.HUGGINGFACE_INFERENCE.value: (
        "intergrax.model_inference.adapters.huggingface_inference_vision",
        "HuggingFaceInferenceVisionAdapter",
    ),
}


class VisionAdapterRegistry:
    """Create vision adapters by provider slug (same pattern as ``LLMAdapterRegistry``)."""

    _factories: dict[str, Callable[..., VisionInferenceAdapter]] = {}

    @staticmethod
    def _normalize_provider(provider: str | VisionProvider) -> str:
        if isinstance(provider, VisionProvider):
            key = provider.value
        elif isinstance(provider, str) and provider.strip():
            key = provider.strip().lower()
        else:
            raise ValueError("provider must be a non-empty VisionProvider or string slug")
        return key

    @classmethod
    def _ensure_builtin(cls, key: str) -> None:
        if key in cls._factories:
            return
        spec = _BUILTIN_VISION_ADAPTERS.get(key)
        if spec is None:
            return
        module_path, class_name = spec
        module = importlib.import_module(module_path)
        adapter_cls = module.__dict__[class_name]

        def factory(**kwargs: Any) -> VisionInferenceAdapter:
            return adapter_cls(**kwargs)

        cls._factories[key] = factory

    @classmethod
    def register(
        cls,
        provider: str | VisionProvider,
        factory: Callable[..., VisionInferenceAdapter],
        *,
        override: bool = False,
    ) -> None:
        key = cls._normalize_provider(provider)
        if key in cls._factories and not override:
            raise ValueError(f"Vision adapter already registered for provider='{key}'")
        cls._factories[key] = factory

    @classmethod
    def create(cls, provider: str | VisionProvider, **kwargs: Any) -> VisionInferenceAdapter:
        key = cls._normalize_provider(provider)
        cls._ensure_builtin(key)
        if key not in cls._factories:
            raise ValueError(f"Vision adapter not registered for provider='{key}'")
        adapter = cls._factories[key](**kwargs)
        if not isinstance(adapter, VisionInferenceAdapter):
            raise TypeError(
                f"Factory for provider='{key}' returned {type(adapter)!r}, expected VisionInferenceAdapter"
            )
        return adapter

    @classmethod
    def registered_providers(cls) -> list[str]:
        keys = set(cls._factories.keys()) | set(_BUILTIN_VISION_ADAPTERS.keys())
        return sorted(keys)
