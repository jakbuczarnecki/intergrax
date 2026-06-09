# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.contracts.autonomy_level import AutonomyLevel
from intergrax.runtime.policy.autonomy_resolver import (
    resolve_effective_autonomy,
    tool_allowed_for_autonomy,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_resolve_effective_autonomy_strict_caps_autonomous() -> None:
    level = resolve_effective_autonomy(
        requested=AutonomyLevel.AUTONOMOUS,
        execution_mode=ExecutionMode.STRICT,
    )
    assert level is AutonomyLevel.ASK


def test_manual_blocks_side_effect_tool() -> None:
    allowed, reason = tool_allowed_for_autonomy("jira.add_comment", AutonomyLevel.MANUAL)
    assert allowed is False
    assert reason == "manual_requires_explicit_approval"


def test_autonomous_allows_side_effect_tool() -> None:
    allowed, _ = tool_allowed_for_autonomy("jira.add_comment", AutonomyLevel.AUTONOMOUS)
    assert allowed is True
