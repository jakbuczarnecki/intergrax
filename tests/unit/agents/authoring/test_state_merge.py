# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.authoring.state_merge import (
    apply_state_delta,
    bump_state_version,
    merge_session_state,
)
from intergrax.contracts.agent_run_enums import AgentRunErrorCode


@pytest.mark.unit
@pytest.mark.gate
def test_apply_state_delta_merge_and_delete() -> None:
    current = {"phase": "plan", "cursor": 1, "scratch": "x"}
    delta = {"phase": "execute", "cursor": 2, "scratch": None}
    merged = apply_state_delta(current, delta)
    assert merged == {"phase": "execute", "cursor": 2}


@pytest.mark.unit
@pytest.mark.gate
def test_bump_state_version_increments() -> None:
    assert bump_state_version({"_version": 3})["_version"] == 4


@pytest.mark.unit
@pytest.mark.gate
def test_merge_session_state_applies_delta_and_version() -> None:
    root = {"acp.state.v1": {"schema_version": "acp.state.v1", "_version": 1, "phase": "plan"}}
    result = merge_session_state(root, {"phase": "execute"})
    assert result.error_code is None
    blob = result.state["acp.state.v1"]
    assert blob["phase"] == "execute"
    assert blob["_version"] == 2


@pytest.mark.unit
@pytest.mark.gate
def test_merge_session_state_rejects_stale_resume() -> None:
    root = {"acp.state.v1": {"schema_version": "acp.state.v1", "_version": 5}}
    result = merge_session_state(root, {"phase": "x"}, incoming_version=3)
    assert result.error_code == AgentRunErrorCode.VALIDATION_FAILED
