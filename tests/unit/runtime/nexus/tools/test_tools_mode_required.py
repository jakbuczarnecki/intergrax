# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-8 / TOOL-ENG-12 — tools governance tests."""

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.config_types import ToolChoiceMode
from intergrax.runtime.nexus.errors.tools_required_error import ToolsRequiredError
from intergrax.runtime.nexus.tools.tool_planning_policy import tool_choice_for_mode

pytestmark = pytest.mark.unit


def test_tool_choice_for_required_mode() -> None:
    assert tool_choice_for_mode("required") == "required"


def test_tool_choice_for_auto_mode() -> None:
    assert tool_choice_for_mode("auto") == "auto"


def test_tools_required_error_message() -> None:
    err = ToolsRequiredError(run_id="run-1")
    assert "run-1" in str(err)
