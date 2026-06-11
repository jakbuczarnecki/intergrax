# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.agent_run_enums import StepNextAction, TerminalReason
from intergrax.contracts.agent_run_trace import (
    AgentRunTrace,
    AgentStepRecord,
    AgentStepStatus,
    GatewayCallStatus,
    LlmCallRecord,
)


@pytest.mark.unit
@pytest.mark.gate
def test_agent_step_record_typed_roundtrip() -> None:
    record = AgentStepRecord(
        step_id="step-0000",
        step_index=0,
        status=AgentStepStatus.SUCCEEDED,
        next_action=StepNextAction.CONTINUE,
        state_version=1,
        llm_calls=[
            LlmCallRecord(
                call_id="llm-1",
                model_id="balanced",
                provider="stub",
                status=GatewayCallStatus.SUCCEEDED,
                tokens_in=10,
                tokens_out=5,
            )
        ],
        terminal_reason=TerminalReason.GOAL_MET,
    )
    restored = AgentStepRecord.model_validate(record.model_dump(mode="json"))
    assert restored.llm_calls[0].model_id == "balanced"
    assert restored.status == AgentStepStatus.SUCCEEDED


@pytest.mark.unit
@pytest.mark.gate
def test_agent_run_trace_rejects_extra_fields() -> None:
    with pytest.raises(ValueError):
        AgentRunTrace.model_validate(
            {"schema_version": "agent_run_trace.v1", "unknown": True},
        )
