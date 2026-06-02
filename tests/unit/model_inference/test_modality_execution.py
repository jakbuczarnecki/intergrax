from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from intergrax.model_inference.contracts import ModelArtifact, ModelArtifactFormat, VisionInferenceRequest
from intergrax.model_inference.execution.in_process_executor import InProcessModalityInferenceExecutor
from intergrax.model_inference.execution.profile import ModalityExecutionMode, ModalityExecutionProfile
from intergrax.model_inference.execution.thread_pool_executor import ThreadPoolModalityInferenceExecutor
from intergrax.model_inference.registry import ModelInferenceRegistry
from intergrax.tools.providers.vision.inference_support import assert_media_within_limit
from intergrax.runtime.modality.modality_profile import ModalityProfile


def test_thread_pool_offloads_heavy_slug_only() -> None:
    registry = ModelInferenceRegistry()
    light = MagicMock()
    light.slug = "onnxruntime"
    heavy = MagicMock()
    heavy.slug = "yolo_ultralytics"
    artifact = ModelArtifact(
        artifact_id="a",
        slug="onnxruntime",
        format=ModelArtifactFormat.ONNX,
    )
    profile = ModalityExecutionProfile(mode=ModalityExecutionMode.THREAD_POOL, max_workers=2)
    pool = ThreadPoolExecutor(max_workers=2)
    executor = ThreadPoolModalityInferenceExecutor(
        profile=profile,
        delegate=InProcessModalityInferenceExecutor(),
        pool=pool,
    )
    request = VisionInferenceRequest(
        request_id="r1",
        artifact_id="a",
        media_uri="file:///tmp/x.png",
    )
    light.detect.return_value = MagicMock(request_id="r1", detections=[])
    heavy.detect.return_value = MagicMock(request_id="r2", detections=[])
    executor.run_detect(registry=registry, adapter=light, artifact=artifact, request=request)
    light.detect.assert_called_once()
    executor.run_detect(
        registry=registry,
        adapter=heavy,
        artifact=artifact,
        request=VisionInferenceRequest(
            request_id="r2",
            artifact_id="a",
            media_uri="file:///tmp/x.png",
        ),
    )
    heavy.detect.assert_called_once()
    pool.shutdown(wait=True)


def test_assert_media_within_limit_rejects_large_file(tmp_path) -> None:
    target = tmp_path / "big.bin"
    target.write_bytes(b"x" * 200)
    profile = ModalityProfile(profile_id="t", max_media_bytes=100)
    with pytest.raises(ValueError, match="max_media_bytes"):
        assert_media_within_limit(profile, target.resolve().as_uri())
