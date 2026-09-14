# © Artur Czarnecki. All rights reserved.

"""EE-B3-C — AC-11 confused deputy (provider must not authorize caller)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.contracts.agent_runtime_governance import CapabilityGrant
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    bind_active_execution_identity,
    reset_active_execution_identity,
)
from intergrax.runtime.agent_governance.approval_boundary import (
    AgentRuntimeApprovalBoundary,
    InMemoryApprovalStore,
)
from intergrax.runtime.agent_governance.audit import (
    GovernanceAuditRecorder,
    InMemoryGovernanceAuditSink,
)
from intergrax.runtime.agent_governance.authorization_boundary import (
    AgentRuntimeGovernanceBoundary,
)
from intergrax.runtime.agent_governance.capability_resolver import (
    InMemoryCapabilityGrantResolver,
)
from intergrax.runtime.agent_governance.errors import ToolGovernanceDeniedError
from intergrax.runtime.agent_governance.pipeline import AgentRuntimeGovernancePipeline
from intergrax.runtime.agent_governance.policy_engine import (
    AgentRuntimePolicyEngine,
    DenyCapabilityPolicyProvider,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _EchoInput(BaseModel):
    value: str


class _EchoOutput(BaseModel):
    value: str


class _CountingExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ToolExecutionRequest[_EchoInput]) -> _EchoOutput:
        self.calls += 1
        return _EchoOutput(value="executed")


class _ProductionConfig:
    policy_bundle = None
    production_mode = True


class _ProductionContext:
    config = _ProductionConfig()


class _ProductionState:
    run_id = "run_01234567890123456789012345678901"
    task_id = "task_01234567890123456789012345678901"
    tenant_id = "tenant-a"
    declarative_hitl_grant = None
    context = _ProductionContext()

    def trace_event(self, **kwargs: object) -> None:
        del kwargs


def _governance_deny_boundary() -> AgentRuntimeGovernanceBoundary:
    grant = CapabilityGrant(
        agent_id="low-priv-agent",
        tenant_id="tenant-a",
        allowed_capabilities=frozenset({"read_invoice"}),
    )
    pipeline = AgentRuntimeGovernancePipeline(
        capability_resolver=InMemoryCapabilityGrantResolver((grant,)),
        policy_engine=AgentRuntimePolicyEngine(
            (
                DenyCapabilityPolicyProvider(
                    denied_capabilities=frozenset({"read_invoice"})
                ),
            ),
        ),
        audit_recorder=GovernanceAuditRecorder(InMemoryGovernanceAuditSink()),
        approval_boundary=AgentRuntimeApprovalBoundary(InMemoryApprovalStore()),
    )
    return AgentRuntimeGovernanceBoundary(pipeline)


def _contract() -> ToolContract:
    return ToolContract(
        tool_id="invoice.read",
        name="Read",
        description="read",
        input_schema=_EchoInput,
        output_schema=_EchoOutput,
        error_mapping={},
        side_effects=False,
        category="read_invoice",
    )


def test_ee_b3_c_low_privilege_caller_cannot_drive_provider_execution() -> None:
    executor = _CountingExecutor()
    contract = _contract()

    class _Handler:
        def execute(self, request: ToolExecutionRequest[_EchoInput]) -> _EchoOutput:
            return _EchoOutput(value="ok")

    registry = ToolRegistry()
    registry.register(contract, _Handler())  # type: ignore[arg-type]
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=executor,  # type: ignore[arg-type]
        agent_runtime_governance=_governance_deny_boundary(),
    )
    token = bind_active_execution_identity(
        run_id=RunId(_ProductionState.run_id),
        attempt_id=AttemptId("attempt_01234567890123456789012345678901"),
        execution_id=ExecutionId("exec_01234567890123456789012345678901"),
    )
    request = ToolExecutionRequest(
        tool_id="invoice.read",
        input=_EchoInput(value="x"),
        run_id=_ProductionState.run_id,
        step_id="step-1",
    )
    try:
        with pytest.raises(ToolGovernanceDeniedError):
            invoker.invoke(
                state=_ProductionState(),  # type: ignore[arg-type]
                agent_id="low-priv-agent",
                request=request,  # type: ignore[arg-type]
            )
    finally:
        reset_active_execution_identity(token)
    assert executor.calls == 0
