# © Artur Czarnecki. All rights reserved.

import pytest
from pydantic import ValidationError

from intergrax.contracts.agent_run import (
    AgentRunCost,
    AgentRunError,
    AgentRunRequest,
    AgentRunResult,
    AgentRunTrace,
    RequestIdentity,
    require_user_id_for_user_memory_scope,
)
from intergrax.contracts.agent_run_enums import (
    AgentRunErrorCode,
    AgentRunStatus,
    PrincipalType,
    TerminalReason,
)


@pytest.mark.unit
@pytest.mark.gate
def test_agent_run_request_roundtrip_json() -> None:
    request = AgentRunRequest(
        input={"query": "hello"},
        identity=RequestIdentity(
            tenant_id="tenant-a",
            user_id="user-1",
            principal_type=PrincipalType.USER,
        ),
        metadata={"locale": "pl"},
        state={"acp.state.v1": {"schema_version": "acp.state.v1", "_version": 1}},
    )
    restored = AgentRunRequest.model_validate_json(request.model_dump_json())
    assert restored == request


@pytest.mark.unit
@pytest.mark.gate
def test_agent_run_result_requires_terminal_reason() -> None:
    with pytest.raises(ValidationError, match="terminal_reason"):
        AgentRunResult(
            status=AgentRunStatus.SUCCEEDED,
            output="done",
        )


@pytest.mark.unit
@pytest.mark.gate
def test_agent_run_result_failed_requires_structured_errors() -> None:
    with pytest.raises(ValidationError, match="errors"):
        AgentRunResult(
            status=AgentRunStatus.FAILED,
            terminal_reason=TerminalReason.ERROR,
        )


@pytest.mark.unit
@pytest.mark.gate
def test_agent_run_result_roundtrip_with_trace_and_cost() -> None:
    result = AgentRunResult(
        status=AgentRunStatus.SUCCEEDED,
        output={"answer": "ok"},
        terminal_reason=TerminalReason.GOAL_MET,
        trace_id="trace-1",
        run_id="run-1",
        trace=AgentRunTrace(run_id="run-1"),
        cost=AgentRunCost(tokens_in=10, tokens_out=5, total_usd=0.01),
        duration_ms=42,
    )
    restored = AgentRunResult.model_validate_json(result.model_dump_json())
    assert restored.status == AgentRunStatus.SUCCEEDED
    assert restored.terminal_reason == TerminalReason.GOAL_MET


@pytest.mark.unit
@pytest.mark.gate
def test_agent_run_error_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AgentRunError.model_validate(
            {
                "code": AgentRunErrorCode.POLICY_DENIED,
                "message": "denied",
                "extra": True,
            }
        )


@pytest.mark.unit
@pytest.mark.gate
def test_require_user_id_for_user_memory_scope() -> None:
    identity = RequestIdentity(tenant_id="tenant-a", user_id=None)
    with pytest.raises(ValueError, match="user_id"):
        require_user_id_for_user_memory_scope(identity, memory_scope="user")

    require_user_id_for_user_memory_scope(identity, memory_scope="org")
