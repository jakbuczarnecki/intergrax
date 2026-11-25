# © Artur Czarnecki. All rights reserved.

"""FC-3 Part D: vision fail-closed and provenance (NT-FC3-VIS1..VIS11)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from intergrax.model_inference.adapters.huggingface_inference_vision import HuggingFaceInferenceVisionAdapter
from intergrax.model_inference.adapters.stub_vision import StubVisionInferenceAdapter
from intergrax.model_inference.adapters.triton_vision import TritonVisionServingAdapter
from intergrax.model_inference.adapters.ultralytics_vision import UltralyticsVisionInferenceAdapter
from intergrax.model_inference.contracts import (
    AuthorizedLocalMedia,
    ModelArtifact,
    ModelArtifactFormat,
    VisionAdapterCapabilityError,
    VisionBackendUnconfiguredError,
    VisionInferenceRequest,
    VisionResultProvenance,
)
from intergrax.model_inference.execution.jobs import ModalityJobResult, ModalityOcrJob, ModalitySegmentJob
from intergrax.model_inference.registry import VisionProfile, VisionProvider
from intergrax.model_inference.registry.vision_adapter_registry import VisionAdapterRegistry
from intergrax.model_inference.workers.task_runner import run_modality_job
from intergrax.tools.providers.speech.backends import MODEL_INFERENCE_REGISTRY_EXTRA_KEY
from intergrax.tools.providers.vision.contracts import VisionDetectInput
from intergrax.tools.providers.vision.inference_support import as_extended_adapter
from intergrax.tools.providers.vision.service import vision_detect
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


@pytest.fixture()
def artifact() -> ModelArtifact:
    return ModelArtifact(
        artifact_id="vision.test",
        slug="vision_serving",
        format=ModelArtifactFormat.REMOTE,
    )


def _local_request(tmp_path, artifact: ModelArtifact, *, request_id: str = "r1") -> VisionInferenceRequest:
    image = tmp_path / "x.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    return VisionInferenceRequest(
        request_id=request_id,
        artifact_id=artifact.artifact_id,
        media_uri=image.resolve().as_uri(),
    )


def test_nt_fc3_vis1_triton_unconfigured_raises_before_media(artifact: ModelArtifact, tmp_path) -> None:
    request = _local_request(tmp_path, artifact)
    with patch(
        "intergrax.model_inference.adapters.triton_vision.require_remote_egress_bytes",
    ) as media_mock:
        with pytest.raises(VisionBackendUnconfiguredError):
            TritonVisionServingAdapter(base_url="").detect(request, artifact=artifact)
        media_mock.assert_not_called()


def test_nt_fc3_vis2_hf_unconfigured_raises_before_media(artifact: ModelArtifact, tmp_path) -> None:
    request = _local_request(tmp_path, artifact, request_id="r2")
    with patch(
        "intergrax.model_inference.adapters.huggingface_inference_vision.require_remote_egress_bytes",
    ) as media_mock:
        with pytest.raises(VisionBackendUnconfiguredError):
            HuggingFaceInferenceVisionAdapter(api_key="").detect(request, artifact=artifact)
        media_mock.assert_not_called()


def test_nt_fc3_vis3_triton_zero_detections_empty_no_stub(artifact: ModelArtifact, tmp_path) -> None:
    image = tmp_path / "x.png"
    image.write_bytes(b"data")
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"outputs": []}
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    mock_client.post.return_value = mock_response
    adapter = TritonVisionServingAdapter(base_url="http://triton.local", model_name="yolo")
    authorized = AuthorizedLocalMedia(
        resolved_path=image.resolve(),
        remote_egress_permitted=True,
    )
    with patch("intergrax.model_inference.adapters.triton_vision.httpx.Client", return_value=mock_client):
        result = adapter.detect(
            VisionInferenceRequest(
                request_id="r3",
                artifact_id=artifact.artifact_id,
                media_uri=image.resolve().as_uri(),
                authorized_local_media=authorized,
            ),
            artifact=artifact,
        )
    assert result.detections == []
    assert result.provenance == VisionResultProvenance.ENGINE


def test_nt_fc3_vis4_hf_zero_detections_empty(artifact: ModelArtifact, tmp_path) -> None:
    image = tmp_path / "x.png"
    image.write_bytes(b"data")
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = []
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    mock_client.post.return_value = mock_response
    adapter = HuggingFaceInferenceVisionAdapter(api_key="hf-test-key")
    authorized = AuthorizedLocalMedia(
        resolved_path=image.resolve(),
        remote_egress_permitted=True,
    )
    with patch(
        "intergrax.model_inference.adapters.huggingface_inference_vision.httpx.Client",
        return_value=mock_client,
    ):
        result = adapter.detect(
            VisionInferenceRequest(
                request_id="r4",
                artifact_id=artifact.artifact_id,
                media_uri=image.resolve().as_uri(),
                authorized_local_media=authorized,
            ),
            artifact=artifact,
        )
    assert result.detections == []
    assert result.provenance == VisionResultProvenance.ENGINE


def test_nt_fc3_vis5_stub_operations_carry_stub_provenance(artifact: ModelArtifact) -> None:
    stub = StubVisionInferenceAdapter()
    request = VisionInferenceRequest(
        request_id="r5",
        artifact_id=artifact.artifact_id,
        media_uri="file:///stub.png",
    )
    detect = stub.detect(request, artifact=artifact)
    segment = stub.segment(request, artifact=artifact)
    ocr = stub.ocr_regions(request, artifact=artifact)
    assert detect.provenance == VisionResultProvenance.STUB
    assert segment.provenance == VisionResultProvenance.STUB
    assert ocr.provenance == VisionResultProvenance.STUB


def test_nt_fc3_vis6_triton_explicit_fallback_flags_stub(artifact: ModelArtifact, tmp_path) -> None:
    request = _local_request(tmp_path, artifact, request_id="r6")
    stub = StubVisionInferenceAdapter()
    adapter = TritonVisionServingAdapter(base_url="", fallback=stub)
    result = adapter.detect(request, artifact=artifact)
    assert result.provenance == VisionResultProvenance.STUB
    assert result.detections[0].label == "object.stub"


def test_nt_fc3_vis7_stub_slug_not_yolo_ultralytics() -> None:
    assert StubVisionInferenceAdapter.slug == "stub_vision"
    assert StubVisionInferenceAdapter.slug != VisionProvider.YOLO_ULTRALYTICS.value
    yolo = VisionAdapterRegistry.create(VisionProvider.YOLO_ULTRALYTICS)
    assert isinstance(yolo, UltralyticsVisionInferenceAdapter)
    assert not isinstance(yolo, StubVisionInferenceAdapter)


def test_nt_fc3_vis8_as_extended_adapter_raises_for_non_extended() -> None:
    adapter = TritonVisionServingAdapter(base_url="http://triton.local")
    with pytest.raises(VisionAdapterCapabilityError):
        as_extended_adapter(adapter)


def test_nt_fc3_vis9_run_modality_job_segment_ocr_errors_without_stub(tmp_path) -> None:
    image = tmp_path / "x.png"
    image.write_bytes(b"data")
    media_uri = image.resolve().as_uri()
    segment_payload = ModalitySegmentJob(
        adapter_slug="vision_serving",
        artifact_id="vision.opencv.onnx",
        request=VisionInferenceRequest(
            request_id="seg1",
            artifact_id="vision.opencv.onnx",
            media_uri=media_uri,
        ),
    ).model_dump_json()
    segment_raw = run_modality_job(segment_payload)
    segment_result = ModalityJobResult.model_validate_json(segment_raw)
    assert segment_result.error
    assert segment_result.segment is None

    ocr_payload = ModalityOcrJob(
        adapter_slug="vision_serving",
        artifact_id="vision.opencv.onnx",
        request=VisionInferenceRequest(
            request_id="ocr1",
            artifact_id="vision.opencv.onnx",
            media_uri=media_uri,
        ),
    ).model_dump_json()
    ocr_raw = run_modality_job(ocr_payload)
    ocr_result = ModalityJobResult.model_validate_json(ocr_raw)
    assert ocr_result.error
    assert ocr_result.ocr is None


def test_nt_fc3_vis10_vision_detect_stub_profile_exposes_provenance(tmp_path) -> None:
    profile = VisionProfile(provider=VisionProvider.STUB)
    registry = profile.build_registry()
    image = tmp_path / "x.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    ctx = ToolWiringContext(
        read_allowlist_roots=frozenset({str(tmp_path.resolve())}),
        extras={MODEL_INFERENCE_REGISTRY_EXTRA_KEY: registry},
    )
    output = vision_detect(
        ctx,
        VisionDetectInput(
            media_uri=image.resolve().as_uri(),
            adapter_slug=StubVisionInferenceAdapter.slug,
            artifact_id=profile.resolved_artifact_id,
        ),
    )
    assert output.provenance == VisionResultProvenance.STUB
    assert output.detections


def test_nt_fc3_vis11_vision_profile_builds_unconfigured_triton_hf(artifact: ModelArtifact, tmp_path) -> None:
    registry = VisionProfile(provider=VisionProvider.OPENCV).build_registry()
    triton = registry.get_vision_adapter(TritonVisionServingAdapter.slug)
    hf = registry.get_vision_adapter(HuggingFaceInferenceVisionAdapter.slug)
    assert isinstance(triton, TritonVisionServingAdapter)
    assert isinstance(hf, HuggingFaceInferenceVisionAdapter)
    request = _local_request(tmp_path, artifact, request_id="r11")
    with pytest.raises(VisionBackendUnconfiguredError):
        triton.detect(request, artifact=artifact)
    with pytest.raises(VisionBackendUnconfiguredError):
        hf.detect(request, artifact=artifact)
