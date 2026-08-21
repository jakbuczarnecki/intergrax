from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from intergrax.model_inference.adapters.huggingface_inference_vision import HuggingFaceInferenceVisionAdapter
from intergrax.model_inference.adapters.triton_vision import TritonVisionServingAdapter
from intergrax.model_inference.bootstrap import build_harness_model_inference_registry
from intergrax.model_inference.opencv_availability import opencv_runtime_available
from intergrax.model_inference.contracts import (
    AuthorizedLocalMedia,
    MediaAuthorizationError,
    ModelArtifact,
    ModelArtifactFormat,
    VisionInferenceRequest,
)
from intergrax.model_inference.media_boundary import resolve_authorized_local_media
from intergrax.runtime.modality.modality_profile import ModalityProfile
from intergrax.tools.providers.speech.backends import MODEL_INFERENCE_REGISTRY_EXTRA_KEY
from intergrax.tools.providers.vision.contracts import VisionDetectInput
from intergrax.tools.providers.vision.inference_support import (
    authorize_vision_media,
    prepare_authorized_vision_media,
)
from intergrax.tools.providers.vision.service import vision_detect
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    root = tmp_path / "sandbox" / "media"
    root.mkdir(parents=True)
    return root


@pytest.fixture()
def allowlisted_ctx(media_root: Path) -> ToolWiringContext:
    return ToolWiringContext(
        read_allowlist_roots=frozenset({str(media_root.resolve())}),
        extras={MODEL_INFERENCE_REGISTRY_EXTRA_KEY: build_harness_model_inference_registry()},
    )


@pytest.fixture()
def remote_artifact() -> ModelArtifact:
    return ModelArtifact(
        artifact_id="vision.test.remote",
        slug="vision_serving",
        format=ModelArtifactFormat.REMOTE,
    )


def _authorized(media_root: Path, relative: str, *, remote_egress: bool) -> AuthorizedLocalMedia:
    return resolve_authorized_local_media(
        relative,
        roots=frozenset({str(media_root.resolve())}),
        remote_egress=remote_egress,
    )


def test_arbitrary_absolute_path_blocked_before_remote_read(
    allowlisted_ctx: ToolWiringContext,
    media_root: Path,
    tmp_path: Path,
) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"TOP_SECRET")
    with patch("intergrax.model_inference.adapters.triton_vision.httpx.Client") as client_cls:
        with pytest.raises(MediaAuthorizationError, match="media_not_in_allowlist"):
            vision_detect(
                allowlisted_ctx,
                VisionDetectInput(
                    media_uri=str(secret.resolve()),
                    adapter_slug="vision_serving",
                    artifact_id="vision.triton.remote",
                ),
            )
        client_cls.assert_not_called()


def test_file_uri_outside_root_blocked(allowlisted_ctx: ToolWiringContext, tmp_path: Path) -> None:
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"data")
    with pytest.raises(MediaAuthorizationError, match="media_not_in_allowlist"):
        authorize_vision_media(
            allowlisted_ctx,
            outside.resolve().as_uri(),
            adapter_slug="vision_serving",
        )


def test_path_traversal_blocked(allowlisted_ctx: ToolWiringContext, media_root: Path, tmp_path: Path) -> None:
    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    (secret_dir / "leak.png").write_bytes(b"data")
    traversal = media_root / ".." / ".." / "secret" / "leak.png"
    with pytest.raises(MediaAuthorizationError, match="media_not_in_allowlist"):
        authorize_vision_media(
            allowlisted_ctx,
            str(traversal),
            adapter_slug="onnxruntime",
        )


@pytest.mark.skipif(os.name == "nt", reason="symlink privilege varies on Windows CI hosts")
def test_symlink_escape_blocked(allowlisted_ctx: ToolWiringContext, media_root: Path, tmp_path: Path) -> None:
    secret = tmp_path / "outside-secret.png"
    secret.write_bytes(b"secret")
    link = media_root / "link.png"
    link.symlink_to(secret)
    with pytest.raises(MediaAuthorizationError, match="media_not_in_allowlist"):
        authorize_vision_media(
            allowlisted_ctx,
            str(link),
            adapter_slug="onnxruntime",
        )


def test_no_roots_fail_closed(media_root: Path) -> None:
    ctx = ToolWiringContext(
        read_allowlist_roots=None,
        extras={MODEL_INFERENCE_REGISTRY_EXTRA_KEY: build_harness_model_inference_registry()},
    )
    allowed = media_root / "ok.png"
    allowed.write_bytes(b"data")
    with patch("intergrax.model_inference.adapters.triton_vision.httpx.Client") as client_cls:
        with pytest.raises(MediaAuthorizationError, match="read_allowlist_not_configured"):
            vision_detect(
                ctx,
                VisionDetectInput(
                    media_uri=str(allowed.resolve()),
                    adapter_slug="vision_serving",
                    artifact_id="vision.triton.remote",
                ),
            )
        client_cls.assert_not_called()


def test_empty_roots_fail_closed(media_root: Path) -> None:
    ctx = ToolWiringContext(
        read_allowlist_roots=frozenset(),
        extras={MODEL_INFERENCE_REGISTRY_EXTRA_KEY: build_harness_model_inference_registry()},
    )
    allowed = media_root / "ok.png"
    allowed.write_bytes(b"data")
    with pytest.raises(MediaAuthorizationError, match="read_allowlist_not_configured"):
        authorize_vision_media(ctx, str(allowed.resolve()), adapter_slug="vision_serving")


def test_authorized_local_file_remote_inference(
    allowlisted_ctx: ToolWiringContext,
    media_root: Path,
    remote_artifact: ModelArtifact,
) -> None:
    payload_bytes = b"authorized-image-bytes"
    image = media_root / "sample.png"
    image.write_bytes(payload_bytes)
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "outputs": [{"label": "person", "confidence": 0.9, "bbox": {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0}}]
    }
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    mock_client.post.return_value = mock_response
    adapter = TritonVisionServingAdapter(base_url="http://triton.local", model_name="yolo")
    authorized = _authorized(media_root, "sample.png", remote_egress=True)
    with patch("intergrax.model_inference.adapters.triton_vision.httpx.Client", return_value=mock_client):
        result = adapter.detect(
            VisionInferenceRequest(
                request_id="ok-remote",
                artifact_id=remote_artifact.artifact_id,
                media_uri="sample.png",
                authorized_local_media=authorized,
            ),
            artifact=remote_artifact,
        )
    assert result.detections[0].label == "person"
    sent_payload = mock_client.post.call_args.kwargs["json"]
    import base64

    assert base64.b64decode(sent_payload["inputs"][0]["data"][0]) == payload_bytes


@pytest.mark.skipif(not opencv_runtime_available(), reason="opencv-python-headless runtime unavailable")
def test_local_adapter_authorized_file() -> None:
    golden = Path(__file__).resolve().parents[4] / "fixtures" / "vision_golden" / "sample_target.png"
    if not golden.is_file():
        pytest.skip("golden vision fixture not present")
    ctx = ToolWiringContext(
        read_allowlist_roots=frozenset({str(golden.parent.resolve())}),
        extras={MODEL_INFERENCE_REGISTRY_EXTRA_KEY: build_harness_model_inference_registry()},
    )
    output = vision_detect(
        ctx,
        VisionDetectInput(media_uri=golden.resolve().as_uri(), adapter_slug="onnxruntime"),
    )
    assert output.detections
    assert output.detections[0].label == "contour.region"


def test_media_size_check_after_authorization(media_root: Path) -> None:
    ctx = ToolWiringContext(read_allowlist_roots=frozenset({str(media_root.resolve())}))
    profile = ModalityProfile(profile_id="bounded", max_media_bytes=4)
    small = media_root / "small.png"
    small.write_bytes(b"12345")
    with pytest.raises(ValueError, match="max_media_bytes"):
        prepare_authorized_vision_media(ctx, profile, str(small.resolve()), adapter_slug="onnxruntime")

    outside = media_root.parent / "outside.png"
    outside.write_bytes(b"999999")
    with patch(
        "intergrax.tools.providers.vision.inference_support.measure_authorized_media_bytes",
    ) as measure:
        with pytest.raises(MediaAuthorizationError, match="media_not_in_allowlist"):
            prepare_authorized_vision_media(ctx, profile, str(outside.resolve()), adapter_slug="onnxruntime")
        measure.assert_not_called()


def test_ordinary_relative_path_within_root_allowed(media_root: Path) -> None:
    image = media_root / "nested" / "view.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"ok")
    authorized = resolve_authorized_local_media(
        "nested/view.png",
        roots=frozenset({str(media_root.resolve())}),
        remote_egress=False,
    )
    assert authorized.resolved_path == image.resolve()


def test_non_local_reference_rejected(allowlisted_ctx: ToolWiringContext) -> None:
    with pytest.raises(MediaAuthorizationError, match="unsupported_media_uri_scheme"):
        authorize_vision_media(
            allowlisted_ctx,
            "https://example.com/image.png",
            adapter_slug="vision_serving",
        )


def test_remote_adapter_rejects_missing_authorized_media(
    remote_artifact: ModelArtifact,
    media_root: Path,
) -> None:
    image = media_root / "x.png"
    image.write_bytes(b"data")
    adapter = HuggingFaceInferenceVisionAdapter(api_key="test-key")
    with patch("intergrax.model_inference.adapters.huggingface_inference_vision.httpx.Client") as client_cls:
        with pytest.raises(MediaAuthorizationError, match="remote_media_egress_not_authorized"):
            adapter.detect(
                VisionInferenceRequest(
                    request_id="missing-auth",
                    artifact_id=remote_artifact.artifact_id,
                    media_uri=image.resolve().as_uri(),
                ),
                artifact=remote_artifact,
            )
        client_cls.assert_not_called()


def test_mod01_vulnerability_path_blocked_no_read_no_http(
    allowlisted_ctx: ToolWiringContext,
    tmp_path: Path,
) -> None:
    """Reproduce audited exfil path: caller local path -> remote adapter."""
    host_secret = Path(os.environ.get("SYSTEMROOT", "C:\\Windows")) / "win.ini"
    if not host_secret.is_file():
        host_secret = tmp_path / "host-readable-secret.bin"
        host_secret.write_bytes(b"HOST_SECRET")
    read_calls: list[Path] = []
    original_read_bytes = Path.read_bytes

    def tracking_read_bytes(self: Path) -> bytes:
        read_calls.append(self)
        return original_read_bytes(self)

    with patch.object(Path, "read_bytes", tracking_read_bytes):
        with patch("intergrax.model_inference.adapters.triton_vision.httpx.Client") as client_cls:
            with pytest.raises(MediaAuthorizationError):
                vision_detect(
                    allowlisted_ctx,
                    VisionDetectInput(
                        media_uri=str(host_secret.resolve()),
                        adapter_slug="vision_serving",
                        artifact_id="vision.triton.remote",
                    ),
                )
            assert not read_calls
            client_cls.assert_not_called()
