# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.6-H1-R1 — verified approval / consumption port coupling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pytest
from pydantic import BaseModel

from intergrax.contracts.agent_governance_approval_consumption_port import (
    AgentGovernanceApprovalConsumptionError,
    AgentGovernanceApprovalConsumptionPort,
)
from intergrax.contracts.agent_governance_verified_approval import (
    VerifiedAgentGovernanceHumanApproval,
)
from intergrax.contracts.agent_runtime_governance import (
    ToolAuthorizationDecision,
    ToolAuthorizationDecisionState,
)
from intergrax.contracts.agent_runtime_policy_evaluation_context import (
    AgentRuntimePolicyEvaluationContext,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    bind_active_execution_identity,
    reset_active_execution_identity,
)
from intergrax.runtime.agent_governance.errors import (
    ToolGovernanceApprovalRequiredError,
    ToolGovernanceDeniedError,
)
from intergrax.runtime.agent_governance.ports import AgentRuntimeGovernancePort
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import RegisteredTool
from intergrax.tools.tool_executor import ToolExecutor
from tests.unit.runtime.architecture.test_uca_6c_r6_r5_foundation_hardening_gates import (
    _grant,
    _pending,
    _requirement,
)

pytestmark = pytest.mark.unit


class _EchoInput(BaseModel):
    value: str


class _EchoOutput(BaseModel):
    value: str


@dataclass
class _OrderLog:
    events: List[str] = field(default_factory=list)


class _AllowGovernance(AgentRuntimeGovernancePort):
    def __init__(self, log: _OrderLog) -> None:
        self._log = log

    def authorize_tool(
        self,
        request: object,
        *,
        policy_context: AgentRuntimePolicyEvaluationContext | None = None,
    ) -> ToolAuthorizationDecision:
        self._log.events.append("governance_allow")
        return ToolAuthorizationDecision(
            decision=ToolAuthorizationDecisionState.ALLOW,
            reason="allowed",
        )


class _DenyGovernance(AgentRuntimeGovernancePort):
    def authorize_tool(
        self,
        request: object,
        *,
        policy_context: AgentRuntimePolicyEvaluationContext | None = None,
    ) -> ToolAuthorizationDecisionState:
        raise ToolGovernanceDeniedError(
            run_id="run_01234567890123456789012345678901",
            agent_id="agent-a",
            tool_id="tool.echo",
            capability="echo",
            reason="denied",
            policy_results=(),
        )


class _RequireApprovalGovernance(AgentRuntimeGovernancePort):
    def authorize_tool(
        self,
        request: object,
        *,
        policy_context: AgentRuntimePolicyEvaluationContext | None = None,
    ) -> ToolAuthorizationDecision:
        raise ToolGovernanceApprovalRequiredError(
            run_id="run_01234567890123456789012345678901",
            agent_id="agent-a",
            tool_id="tool.echo",
            capability="echo",
            approval_id="approval-1",
            reason="still_required",
            policy_results=(),
        )


class _RecordingConsumption(AgentGovernanceApprovalConsumptionPort):
    def __init__(self, log: _OrderLog, *, fail: bool = False) -> None:
        self._log = log
        self._fail = fail

    def mark_applied_after_governance_allow(
        self,
        verified: VerifiedAgentGovernanceHumanApproval,
    ) -> None:
        self._log.events.append("grant_applied")
        if self._fail:
            raise AgentGovernanceApprovalConsumptionError("cas_failed")


class _RecordingExecutor(ToolExecutor):
    def __init__(self, log: _OrderLog) -> None:
        self._log = log

    def execute(self, request: ToolExecutionRequest[BaseModel]) -> _EchoOutput:
        self._log.events.append("backend_start")
        return _EchoOutput(value="ok")


def _verified() -> VerifiedAgentGovernanceHumanApproval:
    requirement = _requirement()
    pending = _pending(requirement)
    grant = _grant(requirement, pending)
    return VerifiedAgentGovernanceHumanApproval(
        verification_id="vagr_test",
        grant_id=grant.grant_id,
        agent_governance_invocation_scope_id=grant.agent_governance_invocation_scope_id,
        requirement=requirement,
        grant=grant,
        logical_invocation_fingerprint=grant.logical_invocation_fingerprint,
        task_id=grant.task_id,
        run_id=grant.run_id,
        attempt_id=grant.attempt_id,
        execution_id=grant.execution_id,
        pause_generation=1,
    )


class _FakeState:
    run_id = "run_01234567890123456789012345678901"
    task_id = "task_01234567890123456789012345678901"
    tenant_id = "tenant-a"
    declarative_hitl_grant = None
    context: object

    def __init__(
        self,
        *,
        verified: VerifiedAgentGovernanceHumanApproval | None,
        consumption: AgentGovernanceApprovalConsumptionPort | None,
    ) -> None:
        self.verified_agent_governance_human_approval = verified
        self.agent_governance_approval_consumption = consumption
        self.context = type(
            "Cfg",
            (),
            {"config": type("PM", (), {"production_mode": False})()},
        )()

    def trace_event(self, **kwargs: object) -> None:
        pass


def _contract() -> ToolContract:
    return ToolContract(
        tool_id="tool.echo",
        name="Echo",
        description="echo",
        input_schema=_EchoInput,
        output_schema=_EchoOutput,
        error_mapping={},
        side_effects=False,
        category="echo",
    )


def _invoke_governance_boundary(
    governance: AgentRuntimeGovernancePort,
    state: _FakeState,
    *,
    log: _OrderLog | None = None,
) -> _OrderLog:
    log = log or _OrderLog()
    contract = _contract()
    registry = type(
        "R",
        (),
        {
            "get": lambda _self, _tid: RegisteredTool(
                contract=contract, handler=lambda req: _EchoOutput(value="x")
            )
        },
    )()
    invoker = RuntimeToolInvoker(
        registry=registry,  # type: ignore[arg-type]
        executor=_RecordingExecutor(log),
        agent_runtime_governance=governance,
    )
    request = ToolExecutionRequest(
        tool_id="tool.echo",
        input=_EchoInput(value="x"),
        run_id=_FakeState.run_id,
        step_id="step-1",
    )
    token = bind_active_execution_identity(
        run_id=RunId(_FakeState.run_id),
        attempt_id=AttemptId("attempt_01234567890123456789012345678901"),
        execution_id=ExecutionId("exec_01234567890123456789012345678901"),
    )
    try:
        invoker._require_agent_runtime_governance(  # noqa: SLF001
            state=state,  # type: ignore[arg-type]
            agent_id="agent-a",
            contract=contract,
            request=request,
        )
        _RecordingExecutor(log).execute(request)
    finally:
        reset_active_execution_identity(token)
    return log


def test_both_none_normal_governance_without_consumption() -> None:
    log = _OrderLog()
    state = _FakeState(verified=None, consumption=None)
    _invoke_governance_boundary(_AllowGovernance(log), state, log=log)
    assert log.events == ["governance_allow", "backend_start"]


def test_both_present_allow_then_applied_then_backend() -> None:
    log = _OrderLog()
    consumption = _RecordingConsumption(log)
    state = _FakeState(verified=_verified(), consumption=consumption)
    _invoke_governance_boundary(_AllowGovernance(log), state, log=log)
    assert log.events.index("governance_allow") < log.events.index("grant_applied")
    assert log.events.index("grant_applied") < log.events.index("backend_start")


def test_verified_without_consumption_fail_closed() -> None:
    log = _OrderLog()
    state = _FakeState(verified=_verified(), consumption=None)
    with pytest.raises(ToolGovernanceDeniedError) as exc_info:
        _invoke_governance_boundary(_AllowGovernance(log), state, log=log)
    assert exc_info.value.reason == "agent_governance_approval_consumption_missing"
    assert "governance_allow" not in log.events
    assert "grant_applied" not in log.events
    assert "backend_start" not in log.events


def test_consumption_without_verified_fail_closed() -> None:
    log = _OrderLog()
    consumption = _RecordingConsumption(log)
    state = _FakeState(verified=None, consumption=consumption)
    with pytest.raises(ToolGovernanceDeniedError) as exc_info:
        _invoke_governance_boundary(_AllowGovernance(log), state, log=log)
    assert exc_info.value.reason == "agent_governance_verified_approval_missing"
    assert "governance_allow" not in log.events
    assert "grant_applied" not in log.events
    assert "backend_start" not in log.events


def test_both_present_deny_skips_applied_and_backend() -> None:
    log = _OrderLog()
    consumption = _RecordingConsumption(log)
    state = _FakeState(verified=_verified(), consumption=consumption)
    with pytest.raises(ToolGovernanceDeniedError):
        _invoke_governance_boundary(_DenyGovernance(), state, log=log)
    assert "grant_applied" not in log.events
    assert "backend_start" not in log.events


def test_both_present_applied_failure_blocks_backend() -> None:
    log = _OrderLog()
    consumption = _RecordingConsumption(log, fail=True)
    state = _FakeState(verified=_verified(), consumption=consumption)
    with pytest.raises(ToolGovernanceDeniedError):
        _invoke_governance_boundary(_AllowGovernance(log), state, log=log)
    assert "governance_allow" in log.events
    assert "grant_applied" in log.events
    assert "backend_start" not in log.events


def test_both_present_require_approval_skips_applied_and_backend() -> None:
    log = _OrderLog()
    consumption = _RecordingConsumption(log)
    state = _FakeState(verified=_verified(), consumption=consumption)
    with pytest.raises(ToolGovernanceApprovalRequiredError):
        _invoke_governance_boundary(_RequireApprovalGovernance(), state, log=log)
    assert "grant_applied" not in log.events
    assert "backend_start" not in log.events
