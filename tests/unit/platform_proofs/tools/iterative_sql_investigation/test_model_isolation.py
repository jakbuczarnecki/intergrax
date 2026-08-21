# © Artur Czarnecki. All rights reserved.

"""Unit tests for model ground-truth isolation in proof prompts."""

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.tools.tool_planning_config import ToolPlanningConfig
from platform_proofs.tools.iterative_sql_investigation.model_context import (
    FORBIDDEN_PROMPT_SUBSTRINGS,
    build_investigation_messages,
)
from platform_proofs.tools.iterative_sql_investigation.scenarios import SCENARIO_A

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
