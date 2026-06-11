# © Artur Czarnecki. All rights reserved.

import pytest
from pydantic import ValidationError

from intergrax.contracts.acp_state import ACP_STATE_KEY, AcpBudgetState, AcpSessionState
from intergrax.contracts.agent_run_enums import CognitivePattern


@pytest.mark.unit
@pytest.mark.gate
def test_acp_session_state_roundtrip() -> None:
    state = AcpSessionState(
        state_version=2,
        pattern=CognitivePattern.REACT,
        phase="act",
        iteration=3,
        budget=AcpBudgetState(steps_used=2, tool_calls=1),
    )
    restored = AcpSessionState.model_validate_json(state.model_dump_json())
    assert restored == state
    assert restored.schema_version == "acp.state.v1"


@pytest.mark.unit
@pytest.mark.gate
def test_acp_session_state_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        AcpSessionState.model_validate(
            {"schema_version": "acp.state.v1", "unknown": True},
        )


@pytest.mark.unit
@pytest.mark.gate
def test_acp_state_key_constant() -> None:
    assert ACP_STATE_KEY == "acp.state.v1"
