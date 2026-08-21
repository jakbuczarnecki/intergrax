from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from intergrax.model_inference.adapters.huggingface_inference_vision import HuggingFaceInferenceVisionAdapter
from intergrax.model_inference.adapters.triton_vision import TritonVisionServingAdapter
from intergrax.model_inference.contracts import AuthorizedLocalMedia, ModelArtifact, ModelArtifactFormat, VisionInferenceRequest


@pytest.fixture()
def artifact() -> ModelArtifact:
    return ModelArtifact(
        artifact_id="vision.test",
        slug="vision_serving",
        format=ModelArtifactFormat.REMOTE,
    )


def test_triton_without_url_falls_back_to_stub(artifact: ModelArtifact, tmp_path) -> None:
    image = tmp_path / "x.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    adapter = TritonVisionServingAdapter(base_url="")
    result = adapter.detect(
        VisionInferenceRequest(
            request_id="r1",
            artifact_id=artifact.artifact_id,
            media_uri=image.resolve().as_uri(),
        ),
        artifact=artifact,
    )
    assert result.detections[0].label == "object.stub"


def test_triton_http_parses_detections(artifact: ModelArtifact, tmp_path) -> None:
    image = tmp_path / "x.png"
    image.write_bytes(b"data")
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "outputs": [{"label": "person", "confidence": 0.88, "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.7}}]
    }
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
                request_id="r2",
                artifact_id=artifact.artifact_id,
                media_uri=image.resolve().as_uri(),
                authorized_local_media=authorized,
            ),
            artifact=artifact,
        )
    assert result.detections[0].label == "person"


def test_hf_without_key_falls_back(artifact: ModelArtifact, tmp_path) -> None:
    image = tmp_path / "x.png"
    image.write_bytes(b"data")
    adapter = HuggingFaceInferenceVisionAdapter(api_key="")
    result = adapter.detect(
        VisionInferenceRequest(
            request_id="r3",
            artifact_id=artifact.artifact_id,
            media_uri=image.resolve().as_uri(),
        ),
        artifact=artifact,
    )
    assert result.detections[0].label == "object.stub"
