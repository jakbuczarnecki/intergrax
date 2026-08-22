# © Artur Czarnecki. All rights reserved.

"""Unit tests for model ground-truth isolation in proof prompts."""

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.tools.tool_planning_config import ToolPlanningConfig
from platform_proofs.tools.iterative_sql_investigation.model_context import (
    FORBIDDEN_PROMPT_SUBSTRINGS,
    build_investigation_messages,
)
from platform_proofs.tools.iterative_sql_investigation.scenarios import SCENARIO_A, SCENARIO_B

pytestmark = pytest.mark.unit


def test_investigation_messages_exclude_hidden_ground_truth() -> None:
    messages = build_investigation_messages(question=SCENARIO_A.question)
    combined = "\n".join(message.content or "" for message in messages)
    for forbidden in FORBIDDEN_PROMPT_SUBSTRINGS:
        assert forbidden not in combined


def test_investigation_policy_is_not_generator_text() -> None:
    policy = ToolPlanningConfig.default().investigation_instructions
    for forbidden in FORBIDDEN_PROMPT_SUBSTRINGS:
        assert forbidden not in policy


def test_scenario_b_prompt_requests_investigation_without_prescribing_controls() -> None:
    messages = build_investigation_messages(question=SCENARIO_B.question)
    user = next(message.content for message in messages if message.role == "user").lower()
    assert "investigate" in user
    assert "observable segment" in user or "relevant observable segments" in user
    assert "correlation" in user and "causation" in user
    for forbidden in (
        "service_type",
        "route_type",
        "2 sql",
        "two sql",
        ">=2",
        "confounded",
    ):
        assert forbidden not in user


def test_generic_investigation_context_includes_falsification_discipline() -> None:
    messages = build_investigation_messages(question=SCENARIO_A.question)
    system = next(message.content for message in messages if message.role == "system")
    policy_lines = [
        line
        for line in system.splitlines()
        if "alternative" in line.lower() or "aggregate result" in line.lower()
    ]
    assert policy_lines
    assert any("alternative" in line.lower() for line in policy_lines)
    assert any("aggregate result" in line.lower() for line in policy_lines)
    assert not any("weight" in line.lower() for line in policy_lines)
