# © Artur Czarnecki. All rights reserved.

"""NPSC-4 integration: tool cannot execute without governance when configured."""

from __future__ import annotations

from typing import Mapping, Type

import pytest
from pydantic import BaseModel

from intergrax.contracts.agent_runtime_governance import (
    CapabilityGrant,
    ToolAuthorizationDecisionState,
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
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import RegisteredTool, ToolRegistry
from intergrax.tools.tool_executor import ToolExecutor

pytestmark = [pytest.mark.unit, pytest.mark.integration]


class _EchoInput(BaseModel):
    value: str


class _EchoOutput(BaseModel):
    value: str


class _FakeRegistry:
    def __init__(self, contract: ToolContract) -> None:
        self._contract = contract

    def get(self, tool_id: str) -> RegisteredTool:
        return RegisteredTool(contract=self._contract, handler=lambda req: _EchoOutput(value="ok"))


class _FakeExecutor(ToolExecutor):
    def execute(self, request: ToolExecutionRequest[BaseModel]) -> _EchoOutput:
        return _EchoOutput(value="executed")


class _FakeState:
    run_id = "run_01234567890123456789012345678901"
    task_id = "task_01234567890123456789012345678901"
    tenant_id = "tenant-a"
    declarative_hitl_grant = None

    def trace_event(self, **kwargs: object) -> None:
        pass


def _contract(tool_id: str = "invoice.read") -> ToolContract:
    return ToolContract(
        tool_id=tool_id,
        name="Read Invoice",
        description="Reads invoice",
        input_schema=_EchoInput,
        output_schema=_EchoOutput,
        error_mapping={},
        side_effects=False,
        category="read_invoice",
    )


def _governance_boundary() -> AgentRuntimeGovernanceBoundary:
    grant = CapabilityGrant(
        agent_id="invoice-agent",
        tenant_id="tenant-a",
        allowed_capabilities=frozenset({"read_invoice"}),
    )
    sink = InMemoryGovernanceAuditSink()
    pipeline = AgentRuntimeGovernancePipeline(
        capability_resolver=InMemoryCapabilityGrantResolver((grant,)),
        policy_engine=AgentRuntimePolicyEngine(
            (DenyCapabilityPolicyProvider(denied_capabilities=frozenset({"read_invoice"})),),
        ),
        audit_recorder=GovernanceAuditRecorder(sink),
        approval_boundary=AgentRuntimeApprovalBoundary(InMemoryApprovalStore()),
    )
    return AgentRuntimeGovernanceBoundary(pipeline)


def test_tool_blocked_without_governance_allow() -> None:
    from intergrax.contracts.execution_identity import (
        AttemptId,
        ExecutionId,
        RunId,
        bind_active_execution_identity,
        reset_active_execution_identity,
    )

    contract = _contract()
    invoker = RuntimeToolInvoker(
        registry=_FakeRegistry(contract),  # type: ignore[arg-type]
        executor=_FakeExecutor(),
        agent_runtime_governance=_governance_boundary(),
    )
    request = ToolExecutionRequest(
        tool_id="invoice.read",
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
        with pytest.raises(ToolGovernanceDeniedError):
            invoker._prepare_invocation(  # noqa: SLF001
                state=_FakeState(),  # type: ignore[arg-type]
                agent_id="invoice-agent",
                request=request,
            )
    finally:
        reset_active_execution_identity(token)
