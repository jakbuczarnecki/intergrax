# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-8 / TOOL-ENG-12 — tools governance tests."""

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.config_types import ToolChoiceMode
from intergrax.runtime.nexus.errors.tools_required_error import ToolsRequiredError
from intergrax.runtime.nexus.tools.native_planner_action_context import (
    NativePlannerProtocolConfig,
    NativePlannerProtocolMode,
)
from intergrax.runtime.nexus.tools.tool_planning_policy import (
    native_tool_choice_for_investigation_round,
    tool_choice_for_mode,
)

pytestmark = pytest.mark.unit


def test_tool_choice_for_required_mode() -> None:
    assert tool_choice_for_mode("required") == "required"


def test_tool_choice_for_auto_mode() -> None:
    assert tool_choice_for_mode("auto") == "auto"


def test_tools_required_error_message() -> None:
    err = ToolsRequiredError(run_id="run-1")
    assert "run-1" in str(err)


def test_investigation_round_without_prior_evidence_allows_auto() -> None:
    protocol = NativePlannerProtocolConfig(
        mode=NativePlannerProtocolMode.INVESTIGATION_ATOMIC_ROUND,
    )
    assert native_tool_choice_for_investigation_round(
        protocol_config=protocol,
        tools_mode="auto",
    ) == "auto"


def test_investigation_round_with_prior_evidence_allows_auto_in_final_capable_round() -> None:
    protocol = NativePlannerProtocolConfig(
        mode=NativePlannerProtocolMode.INVESTIGATION_ATOMIC_ROUND,
        available_evidence_references=("obs.ref.a",),
    )
    assert native_tool_choice_for_investigation_round(
        protocol_config=protocol,
        tools_mode="auto",
    ) == "auto"


def test_investigation_round_with_prior_evidence_still_requires_tool_when_tools_mode_required() -> None:
    protocol = NativePlannerProtocolConfig(
        mode=NativePlannerProtocolMode.INVESTIGATION_ATOMIC_ROUND,
        available_evidence_references=("obs.ref.a",),
    )
    assert native_tool_choice_for_investigation_round(
        protocol_config=protocol,
        tools_mode="required",
    ) == "required"
