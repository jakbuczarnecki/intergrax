from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.model_inference.bootstrap import build_harness_model_inference_registry
from intergrax.model_inference.opencv_availability import opencv_runtime_available
from intergrax.tools.providers.ml.contracts import MlPredictInput
from intergrax.tools.providers.ml.service import ml_predict
from intergrax.tools.providers.speech.backends import (
    MODEL_INFERENCE_REGISTRY_EXTRA_KEY,
    SPEECH_BACKEND_EXTRA_KEY,
    StubSpeechBackend,
)
from intergrax.tools.providers.speech.contracts import SpeechSynthesizeInput
from intergrax.tools.providers.speech.service import speech_synthesize
from intergrax.tools.providers.vision.contracts import VisionDetectInput
from intergrax.tools.providers.vision.service import vision_detect
from intergrax.tools.registry.wiring import ToolWiringContext


def _harness_tool_ctx(golden: Path | None = None) -> ToolWiringContext:
    roots: frozenset[str] | None = None
    if golden is not None:
        roots = frozenset({str(golden.parent.resolve())})
    return ToolWiringContext(
        read_allowlist_roots=roots,
        extras={
            SPEECH_BACKEND_EXTRA_KEY: StubSpeechBackend(),
            MODEL_INFERENCE_REGISTRY_EXTRA_KEY: build_harness_model_inference_registry(),
        },
    )


def test_speech_synthesize_stub_returns_audio_uri() -> None:
    output = speech_synthesize(
        _harness_tool_ctx(),
        SpeechSynthesizeInput(text="hello harness"),
    )
    assert output.audio_uri.startswith("stub://speech/")
    assert output.character_count == len("hello harness")


@pytest.mark.skipif(not opencv_runtime_available(), reason="opencv-python-headless runtime unavailable")
def test_vision_detect_opencv_returns_detection() -> None:
    golden = Path(__file__).resolve().parents[3] / "fixtures" / "vision_golden" / "sample_target.png"
    if not golden.is_file():
        pytest.skip("golden vision fixture not present")
    output = vision_detect(
        _harness_tool_ctx(golden),
        VisionDetectInput(media_uri=golden.resolve().as_uri()),
    )
    assert output.detections
    assert output.detections[0].label == "contour.region"


def test_ml_predict_stub_returns_score() -> None:
    output = ml_predict(_harness_tool_ctx(), MlPredictInput(features={"x": 0.8, "y": 0.2}))
    assert "positive" in output.predictions
