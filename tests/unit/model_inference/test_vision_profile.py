from __future__ import annotations

import pytest

from intergrax.model_inference.adapters.opencv_vision import OpenCvVisionInferenceAdapter
from intergrax.model_inference.adapters.stub_vision import StubVisionInferenceAdapter
from intergrax.model_inference.registry import VisionProfile, VisionProvider, vision_profile_from_env
from intergrax.model_inference.registry.vision_adapter_registry import VisionAdapterRegistry


def test_vision_profile_create_adapter_opencv() -> None:
    profile = VisionProfile(provider=VisionProvider.OPENCV)
    adapter = profile.create_adapter()
    assert isinstance(adapter, OpenCvVisionInferenceAdapter)
    assert profile.resolved_artifact_id == "vision.opencv.onnx"


def test_vision_profile_create_adapter_stub() -> None:
    profile = VisionProfile(provider=VisionProvider.STUB)
    adapter = profile.create_adapter()
    assert isinstance(adapter, StubVisionInferenceAdapter)


def test_vision_profile_build_registry_registers_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_VISION_PROVIDER", "stub")
    profile = vision_profile_from_env()
    registry = profile.build_registry()
    adapter = registry.get_vision_adapter(profile.adapter_slug)
    assert isinstance(adapter, StubVisionInferenceAdapter)
    artifact = registry.get_artifact(profile.resolved_artifact_id)
    assert artifact.slug == adapter.slug


def test_vision_profile_from_env_legacy_adapter_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERGRAX_VISION_PROVIDER", raising=False)
    monkeypatch.setenv("INTERGRAX_VISION_ADAPTER", "yolo_ultralytics")
    profile = vision_profile_from_env()
    assert profile.provider == VisionProvider.YOLO_ULTRALYTICS


def test_vision_adapter_registry_lists_builtin_providers() -> None:
    providers = VisionAdapterRegistry.registered_providers()
    assert VisionProvider.OPENCV.value in providers
    assert VisionProvider.STUB.value in providers
