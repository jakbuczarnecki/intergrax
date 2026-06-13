# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from lab_application.host.settings import LabApplicationSettings
from lab_application.host.tool_wiring import wire_lab_tools

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_lab_settings_mcp_default_off() -> None:
    settings = LabApplicationSettings()
    assert settings.include_mcp is False


def test_lab_settings_strict_harness_default_off() -> None:
    settings = LabApplicationSettings()
    assert settings.strict_harness is False


@pytest.mark.no_ci
def test_wire_lab_tools_omits_sandbox_without_session() -> None:
    wiring = wire_lab_tools()
    assert "sandbox.exec" not in wiring.profile.enabled


@pytest.mark.no_ci
def test_wire_lab_tools_includes_sandbox_when_session_wired() -> None:
    wiring = wire_lab_tools(sandbox_session=object())
    assert "sandbox.exec" in wiring.profile.enabled


@pytest.mark.no_ci
def test_wire_lab_tools_harness_enables_modality_tools_and_profile() -> None:
    from intergrax.model_inference.registry.vision_profile import VISION_PROFILE_EXTRA_KEY
    from intergrax.runtime.modality.modality_profile import MODALITY_PROFILE_EXTRA_KEY
    from intergrax.speech_adapters.registry.profile import SPEECH_PROFILE_EXTRA_KEY
    from intergrax.tools.providers.speech.backends import (
        MODEL_INFERENCE_REGISTRY_EXTRA_KEY,
        SPEECH_BACKEND_EXTRA_KEY,
    )

    wiring = wire_lab_tools(harness=True)
    for tool_id in (
        "speech.synthesize",
        "speech.transcribe",
        "vision.detect",
        "vision.segment",
        "vision.ocr_regions",
        "ml.predict",
        "ml.explain",
        "ml.batch_predict",
    ):
        assert tool_id in wiring.profile.enabled
    extras = wiring.wiring_context.extras
    assert MODALITY_PROFILE_EXTRA_KEY in extras
    assert VISION_PROFILE_EXTRA_KEY in extras
    assert SPEECH_PROFILE_EXTRA_KEY in extras
    assert SPEECH_BACKEND_EXTRA_KEY in extras
    assert MODEL_INFERENCE_REGISTRY_EXTRA_KEY in extras
