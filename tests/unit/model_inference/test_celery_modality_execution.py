from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.model_inference.contracts import ModelArtifact, ModelArtifactFormat, VisionInferenceRequest
from intergrax.model_inference.execution.celery_executor import CeleryModalityInferenceExecutor
from intergrax.model_inference.execution.jobs import ModalityDetectJob
from intergrax.model_inference.execution.profile import ModalityExecutionMode, ModalityExecutionProfile
from intergrax.model_inference.registry import ModelInferenceRegistry
from intergrax.model_inference.bootstrap import build_harness_model_inference_registry
from intergrax.model_inference.execution.jobs import ModalityJobResult
from intergrax.model_inference.registry.vision_profile import vision_profile_from_env
from intergrax.model_inference.workers import celery_app as celery_module
from intergrax.model_inference.workers.task_runner import run_modality_job
from intergrax.model_inference.opencv_availability import opencv_runtime_available

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_celery_app() -> None:
    celery_module.reset_modality_celery_app()
    yield
    celery_module.reset_modality_celery_app()


def test_celery_without_broker_falls_back_to_delegate() -> None:
    registry = ModelInferenceRegistry()
    adapter = MagicMock()
    adapter.slug = "yolo_ultralytics"
    artifact = ModelArtifact(
        artifact_id="art-1",
        slug="yolo_ultralytics",
        format=ModelArtifactFormat.ONNX,
    )
    delegate = MagicMock()
    delegate.run_detect.return_value = MagicMock(request_id="fb", detections=[])
    profile = ModalityExecutionProfile(mode=ModalityExecutionMode.CELERY)
    executor = CeleryModalityInferenceExecutor(profile=profile, fallback=delegate)
    request = VisionInferenceRequest(
        request_id="r1",
        artifact_id="art-1",
        media_uri="file:///tmp/x.png",
    )
    result = executor.run_detect(registry=registry, adapter=adapter, artifact=artifact, request=request)
    assert result.request_id == "fb"
    delegate.run_detect.assert_called_once()


@pytest.mark.skipif(not opencv_runtime_available(), reason="opencv-python-headless runtime unavailable")
def test_run_modality_detect_job_uses_harness_registry(vision_golden_image, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_VISION_PROVIDER", "onnxruntime")
    profile = vision_profile_from_env()
    registry = build_harness_model_inference_registry(profile=profile)
    artifact_id = profile.resolved_artifact_id
    adapter_slug = profile.adapter_slug
    registry.get_artifact(artifact_id)
    job = ModalityDetectJob(
        adapter_slug=adapter_slug,
        artifact_id=artifact_id,
        request=VisionInferenceRequest(
            request_id="req-1",
            artifact_id=artifact_id,
            media_uri=vision_golden_image.resolve().as_uri(),
        ),
    )
    raw = run_modality_job(job.model_dump_json())
    result = ModalityJobResult.model_validate_json(raw)
    assert result.error is None
    assert result.detect is not None
