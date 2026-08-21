from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.model_inference.bootstrap import build_harness_model_inference_registry
from intergrax.model_inference.opencv_availability import opencv_runtime_available
from intergrax.tools.providers.ml.contracts import MlExplainInput
from intergrax.tools.providers.ml.service import ml_explain
from intergrax.speech_adapters.providers.stub_speech import StubSpeechAdapter
from intergrax.tools.providers.speech.backends import MODEL_INFERENCE_REGISTRY_EXTRA_KEY, SPEECH_BACKEND_EXTRA_KEY
from intergrax.tools.providers.vision.contracts import VisionSegmentInput
from intergrax.tools.providers.vision.service import vision_segment
from intergrax.tools.registry.wiring import ToolWiringContext

_GOLDEN = Path(__file__).resolve().parents[3] / "fixtures" / "vision_golden" / "sample_target.png"


def _vision_ctx(golden: Path) -> ToolWiringContext:
    return ToolWiringContext(
        read_allowlist_roots=frozenset({str(golden.parent.resolve())}),
        extras={
            MODEL_INFERENCE_REGISTRY_EXTRA_KEY: build_harness_model_inference_registry(),
            SPEECH_BACKEND_EXTRA_KEY: StubSpeechAdapter(),
        },
    )


def _ml_ctx() -> ToolWiringContext:
    return ToolWiringContext(
        extras={
            MODEL_INFERENCE_REGISTRY_EXTRA_KEY: build_harness_model_inference_registry(),
            SPEECH_BACKEND_EXTRA_KEY: StubSpeechAdapter(),
        },
    )


@pytest.mark.skipif(not opencv_runtime_available(), reason="opencv-python-headless runtime unavailable")
def test_vision_segment_returns_contours() -> None:
    if not _GOLDEN.is_file():
        pytest.skip("golden fixture missing")
    output = vision_segment(_vision_ctx(_GOLDEN), VisionSegmentInput(media_uri=_GOLDEN.resolve().as_uri()))
    assert output.segments
    assert output.segments[0].label == "contour.segment"


def test_ml_explain_returns_importance() -> None:
    output = ml_explain(_ml_ctx(), MlExplainInput(features={"a": 1.0, "b": 3.0}))
    assert output.predictions
    assert output.feature_importance["b"] > output.feature_importance["a"]
