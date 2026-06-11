# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.authoring.state_access import load_session_state, session_state_delta
from intergrax.contracts.acp_state import ACP_STATE_KEY, AcpSessionState
from intergrax.contracts.agent_step_context import AgentStepContext


class _ResearchState(AcpSessionState):
    plan_cursor: int = 0


@pytest.mark.unit
@pytest.mark.gate
def test_load_session_state_from_nested_blob() -> None:
    ctx = AgentStepContext(
        state_snapshot={
            ACP_STATE_KEY: {
                "schema_version": "acp.state.v1",
                "_version": 1,
                "plan_cursor": 2,
            }
        }
    )
    state = load_session_state(ctx, state_type=_ResearchState)
    assert isinstance(state, _ResearchState)
    assert state.plan_cursor == 2


@pytest.mark.unit
@pytest.mark.gate
def test_session_state_delta_excludes_envelope_fields() -> None:
    state = _ResearchState(plan_cursor=4, phase="execute")
    delta = session_state_delta(state)
    assert delta == {"plan_cursor": 4, "phase": "execute", "iteration": 0}
    assert "schema_version" not in delta
    assert "_version" not in delta
