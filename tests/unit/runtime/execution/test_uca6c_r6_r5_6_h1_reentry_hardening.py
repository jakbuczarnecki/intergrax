# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.6-H1 — reentry boundary, APPLIED timing, durable reject."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pytest
from pydantic import BaseModel

from intergrax.contracts.agent_governance_approval_consumption_port import (
    AgentGovernanceApprovalConsumptionError,
    AgentGovernanceApprovalConsumptionPort,
)
from intergrax.contracts.agent_decision import HumanRequest
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
from intergrax.runtime.agent_governance.errors import ToolGovernanceDeniedError
from intergrax.runtime.agent_governance.ports import AgentRuntimeGovernancePort
from intergrax.runtime.human.agent_governance_human_approval_grant import (
    AgentGovernanceHumanApprovalGrantCoordinator,
)
from intergrax.runtime.human.agent_governance_pause_projection import (
    AgentGovernancePauseProjectionOutcome,
    TaskAgentGovernancePauseProjectionAdapter,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.task.task import Task
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import RegisteredTool
from intergrax.tools.tool_executor import ToolExecutor
from tests.unit.runtime.architecture.test_uca_6c_r6_r5_foundation_hardening_gates import (
    _grant,
    _pending,
    _requirement,
)
from tests.unit.runtime.human.test_agent_governance_grant_lifecycle_h1 import (
    _MemoryTaskCheckpointStore,
    _seed_checkpoint,
    _task,
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
    agent_governance_approval_consumption = None

    context: object

    def __init__(self, consumption: AgentGovernanceApprovalConsumptionPort) -> None:
        self.verified_agent_governance_human_approval = _verified()
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


def _run_governance_boundary_then_backend(
    governance: AgentRuntimeGovernancePort,
    consumption: AgentGovernanceApprovalConsumptionPort,
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
    executor = _RecordingExecutor(log)
    invoker = RuntimeToolInvoker(
        registry=registry,  # type: ignore[arg-type]
        executor=executor,
        agent_runtime_governance=governance,
    )
    state = _FakeState(consumption)
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
        executor.execute(request)
    finally:
        reset_active_execution_identity(token)
    return log


def test_mark_applied_before_backend_on_governance_allow() -> None:
    log = _OrderLog()
    consumption = _RecordingConsumption(log)
    events = _run_governance_boundary_then_backend(
        _AllowGovernance(log),
        consumption,
        log=log,
    ).events
    assert events.index("governance_allow") < events.index("grant_applied")
    assert events.index("grant_applied") < events.index("backend_start")


def test_deny_skips_applied_and_backend() -> None:
    log = _OrderLog()
    consumption = _RecordingConsumption(log)
    with pytest.raises(ToolGovernanceDeniedError):
        _run_governance_boundary_then_backend(_DenyGovernance(), consumption)
    assert "grant_applied" not in log.events
    assert "backend_start" not in log.events


def test_mark_applied_cas_failure_blocks_backend() -> None:
    log = _OrderLog()
    consumption = _RecordingConsumption(log, fail=True)
    with pytest.raises(ToolGovernanceDeniedError):
        _run_governance_boundary_then_backend(
            _AllowGovernance(log),
            consumption,
            log=log,
        )
    assert "governance_allow" in log.events
    assert "grant_applied" in log.events
    assert "backend_start" not in log.events


def test_reject_pending_cleared_after_checkpoint_reload() -> None:
    task = _task()
    store = _MemoryTaskCheckpointStore()
    _seed_checkpoint(task, store)
    pending = _pending(_requirement())
    task.runtime.governance.agent_governance_hitl_pending = pending
    AgentGovernanceHumanApprovalGrantCoordinator.clear_pending_on_reject_or_escalate(
        task,
        checkpoint_store=store,
    )
    latest = store.get_latest(str(task.task_id), task.tenant_id)
    assert latest is not None
    reloaded = Task.model_validate(latest.task_snapshot)
    assert reloaded.runtime.governance.agent_governance_hitl_pending is None


def test_stale_writer_reject_clear_fails_closed() -> None:
    task = _task()
    store = _MemoryTaskCheckpointStore()
    _seed_checkpoint(task, store)
    requirement = _requirement()
    pending = _pending(requirement)
    adapter_seed = TaskAgentGovernancePauseProjectionAdapter(
        task=task,
        checkpoint_store=store,
    )
    seeded = adapter_seed.persist_pause_projection(
        pending=pending,
        human_request=HumanRequest(request_id=pending.human_request_id, prompt="ok?"),
    )
    assert seeded.outcome is AgentGovernancePauseProjectionOutcome.APPLIED
    base_revision = store.get_latest(str(task.task_id), task.tenant_id)
    assert base_revision is not None
    task_b = Task.model_validate(task.model_dump(mode="json"))
    adapter_a = TaskAgentGovernancePauseProjectionAdapter(
        task=task,
        checkpoint_store=store,
    )
    adapter_b = TaskAgentGovernancePauseProjectionAdapter(
        task=task_b,
        checkpoint_store=store,
    )
    first = adapter_a.clear_pending_durably()
    assert first.outcome is AgentGovernancePauseProjectionOutcome.APPLIED
    stale = adapter_b._clear_pending_through_checkpoint(
        expected_checkpoint_revision=base_revision.revision,
    )
    assert stale.outcome is AgentGovernancePauseProjectionOutcome.STALE_REVISION
