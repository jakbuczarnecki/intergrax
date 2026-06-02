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


def test_wire_lab_tools_omits_sandbox_without_session() -> None:
    wiring = wire_lab_tools()
    assert "sandbox.exec" not in wiring.profile.enabled


def test_wire_lab_tools_includes_sandbox_when_session_wired() -> None:
    wiring = wire_lab_tools(sandbox_session=object())
    assert "sandbox.exec" in wiring.profile.enabled
