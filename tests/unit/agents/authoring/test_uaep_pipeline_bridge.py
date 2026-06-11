# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.authoring.uaep_pipeline_bridge import (
    pipeline_agent_steps,
    pipeline_step_complete,
)
from intergrax.contracts.agent_decision import AgentDecisionType

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_pipeline_agent_steps_builds_single_step() -> None:
    steps = pipeline_agent_steps(step_id="domain", step_name="Domain step", trace_label="domain")
    assert len(steps) == 1
    assert steps[0].step_id == "domain"
    assert steps[0].allowed_tools == []


def test_pipeline_step_complete_returns_complete_decision() -> None:
    decision = pipeline_step_complete(reason="done")
    assert decision.type == AgentDecisionType.COMPLETE
    assert decision.reason == "done"
