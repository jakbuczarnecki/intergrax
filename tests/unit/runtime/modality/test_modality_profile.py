from __future__ import annotations

from intergrax.runtime.modality.modality_profile import (
    ModalityPlane,
    ModalityProfile,
    filter_tool_ids_by_modality_profile,
    lab_default_modality_profile,
)


def test_lab_profile_allows_modality_tools() -> None:
    profile = lab_default_modality_profile()
    filtered = filter_tool_ids_by_modality_profile(
        ("rag.retrieve", "vision.detect", "speech.synthesize", "ml.predict", "notify.send"),
        profile,
    )
    assert "vision.detect" in filtered
    assert "speech.synthesize" in filtered
    assert "ml.predict" in filtered
    assert "notify.send" not in filtered


def test_dedicated_inference_plane_filters_speech_and_vision() -> None:
    profile = ModalityProfile(
        profile_id="inference-only",
        allowed_planes={ModalityPlane.DEDICATED_INFERENCE},
    )
    filtered = filter_tool_ids_by_modality_profile(
        ("vision.detect", "rag.retrieve"),
        profile,
    )
    assert filtered == ("vision.detect",)
