# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-30 — lab host tool invocation mode wiring."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from intergrax.runtime.nexus.config_types import ToolInvocationMode
from lab_application.host.settings import LabApplicationSettings

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.parametrize(
    "mode",
    [mode.value for mode in ToolInvocationMode],
)
def test_lab_environment_profile_accepts_shipped_invocation_modes(mode: str) -> None:
    settings = LabApplicationSettings(tool_invocation_mode=mode)
    env = build_lab_environment_profile(settings)
    assert env.tool_invocation_mode == mode
