from __future__ import annotations

from intergrax.tools.providers.ml.service import ml_predict
from intergrax.tools.providers.ml.contracts import MlPredictInput
from intergrax.tools.providers.speech.contracts import SpeechSynthesizeInput
from intergrax.tools.providers.speech.service import speech_synthesize
from intergrax.tools.providers.vision.contracts import VisionDetectInput
from intergrax.tools.providers.vision.service import vision_detect


def test_speech_synthesize_stub_returns_audio_uri() -> None:
    output = speech_synthesize(SpeechSynthesizeInput(text="hello harness"))
    assert output.audio_uri.startswith("stub://speech/")
    assert output.character_count == len("hello harness")


def test_vision_detect_stub_returns_detection() -> None:
    output = vision_detect(VisionDetectInput(media_uri="file:///tmp/sample.png"))
    assert output.detections
    assert output.detections[0].label == "object.stub"


def test_ml_predict_stub_returns_score() -> None:
    output = ml_predict(MlPredictInput(features={"x": 0.8, "y": 0.2}))
    assert "positive" in output.predictions
