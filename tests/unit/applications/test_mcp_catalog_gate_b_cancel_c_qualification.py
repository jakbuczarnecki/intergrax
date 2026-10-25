# © Artur Czarnecki. All rights reserved.

"""TLX-TL11 Gate B P07 + CANCEL-C — MCP catalog invoke and governed cancellation qualification."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from pydantic import BaseModel

from intergrax.applications._shared.harness_control_plane_policy_wiring import (
    build_harness_control_plane_mutation_boundary,
)
from intergrax.applications._shared.mcp_catalog_invocation_bridge import (
    McpCatalogInvocationComposition,
    _METADATA_STEP_ID,
)
from intergrax.applications._shared.mcp_host_execution_transport_cancel import (
    McpProtectedHostExecutionAdmissionError,
    create_mcp_protected_host_execution_launch,
)
from intergrax.contracts.collaborative_work import CollaborativeWorkEnforcementRequest
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    reset_active_execution_identity,
)
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.contracts.root_execution_transport_detachment_supervision import (
    RootExecutionTransportAdmissionRequest,
    RootExecutionTransportDetachmentHandoffDisposition,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.execution.root_execution_transport_detachment_supervision import (
    build_host_root_execution_transport_detachment_supervisor,
)
from intergrax.applications._shared.mcp_harness_principal import McpRunAgentIdentityResolution
from intergrax.applications._shared.mcp_nexus_server import build_nexus_mcp_server
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
)
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.contracts.tool_execution_outcome import ToolEffectCertainty, ToolExecutionResult
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.tool_executor import ToolHandler

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-gate-b"
_OTHER_TENANT = "tenant-other"
_USER = "operator-gate-b"
_OTHER_USER = "intruder-gate-b"
_STEP = "gate-b.catalog.step"


class _In(BaseModel):
    query: str = "x"


class _Out(BaseModel):
    value: str = "ok"


@dataclass
class _RecordingCatalogInvoker:
    caller_agent_id: str = "agent-gate-b"
    results: list[ExecutionBoundCatalogToolInvokeRequest] = field(default_factory=list)
    delay_s: float = 0.0
    fail_code: str | None = None
    uncertain: bool = False
    started: threading.Event = field(default_factory=threading.Event)
    completed: threading.Event = field(default_factory=threading.Event)

    def invoke(self, request: ExecutionBoundCatalogToolInvokeRequest) -> ToolExecutionResult[_Out]:
        self.results.append(request)
        self.started.set()
        if self.delay_s > 0:
            time.sleep(self.delay_s)
        if CancellationCoordinator.is_requested({}):
            pass
        if self.fail_code:
            return ToolExecutionResult.fail(self.fail_code, "injected failure")
        if self.uncertain:
            self.completed.set()
            return ToolExecutionResult(
                success=False,
                output=None,
                error=None,
                effect_certainty=ToolEffectCertainty.UNCERTAIN,
            )
        result = ToolExecutionResult.ok(_Out(value="invoked"))
        self.completed.set()
        return result


class _Enablement:
    def __init__(self, enabled: set[str]) -> None:
        self._enabled = enabled

    def is_tool_enabled(self, tool_id: str) -> bool:
        return tool_id in self._enabled


def _contract(
    tool_id: str,
    *,
    side_effects: bool = False,
) -> ToolContract:
    return ToolContract(
        tool_id=tool_id,
        name=tool_id,
        description="gate-b qual",
        input_schema=_In,
        output_schema=_Out,
        error_mapping={},
        side_effects=side_effects,
        risk_level=ToolRiskLevel.LOW,
        category="gate-b",
    )


def _handler() -> ToolHandler[_In, _Out]:
    async def _run(_input: _In, _ctx: object) -> _Out:
        return _Out(value="should-not-run")

    return _run


def _principal(*, tenant: str = _TENANT, user_id: str = _USER) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant,
        user_id=user_id,
        principal_type=PrincipalType.USER,
        auth_subject=user_id,
    )


def _identity_resolution(
    *,
    tenant: str = _TENANT,
    user_id: str = _USER,
) -> McpRunAgentIdentityResolution:
    principal = _principal(tenant=tenant, user_id=user_id)
    return McpRunAgentIdentityResolution(
        tenant_id=tenant,
        user_id=user_id,
        principal=None,
        request_identity=principal,
    )


async def _register_binding(
    *,
    tenant: str = _TENANT,
    user_id: str = _USER,
    step_id: str = _STEP,
    agent_id: str = "agent-gate-b",
) -> tuple[Task, str]:
    task_id = mint_task_id()
    run_id = mint_run_id()
    metadata: dict[str, Any] = {_METADATA_STEP_ID: step_id}
    task = Task(
        task_id=task_id,
        tenant_id=tenant,
        user_id=user_id,
        message="gate-b",
        context=TaskContext(capability="gate-b.cap"),
        metadata=metadata,
        agent_id=agent_id,
    )
    await ActiveTaskRegistry.register(task, run_id)
    return task, str(run_id)


def _build_mcp(
    registry: ToolRegistry,
    invoker: _RecordingCatalogInvoker,
    *,
    enablement: _Enablement | None = None,
    with_boundary: bool = True,
    auto_wire_catalog_transport_supervision: bool = True,
    transport_detachment_supervision: object | None = None,
) -> object:
    boundary = None
    if with_boundary:
        env = ApplicationEnvironmentProfile(application_profile=ApplicationProfile.PRODUCT)
        boundary = build_harness_control_plane_mutation_boundary(env)
    return build_nexus_mcp_server(
        name="gate-b-cancel-c",
        host_execution=AsyncMock(spec=HostTaskExecutionPort),
        registry=AgentRegistry(),
        default_capability="gate-b.cap",
        default_tenant_id=_TENANT,
        default_user_id=_USER,
        tool_registry=registry,
        catalog_tool_invoker=invoker,
        tool_enablement=enablement,
        task_control_mutation_boundary=boundary,
        auto_wire_catalog_transport_supervision=auto_wire_catalog_transport_supervision,
        transport_detachment_supervision=transport_detachment_supervision,
    )


@pytest.fixture(autouse=True)
async def _clear_active_tasks() -> None:
    ActiveTaskRegistry.clear_for_tests()
    yield
    ActiveTaskRegistry.clear_for_tests()


@pytest.fixture(autouse=True)
def _fast_mcp_transport_drain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "intergrax.applications._shared.mcp_host_execution_transport_cancel."
        "_HOST_DRAIN_AFTER_TRANSPORT_CANCEL_S",
        0.25,
    )


@pytest.mark.asyncio
async def test_gate_b_successful_authorized_invoke() -> None:
    """External MCP HTTP catalog invoke is disabled until cross-loop ingress (F3)."""
    registry = ToolRegistry()
    registry.register(_contract("gate.b.ok"), _handler())
    invoker = _RecordingCatalogInvoker()
    mcp = _build_mcp(registry, invoker)
    task, run_id = await _register_binding()
    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
        return_value=_identity_resolution(),
    ):
        async with Client(mcp) as client:
            result = await client.call_tool(
                "invoke_catalog_tool",
                {
                    "tool_id": "gate.b.ok",
                    "task_id": str(task.task_id),
                    "run_id": run_id,
                    "step_id": _STEP,
                    "input": {"query": "hi"},
                },
            )
    assert result.data["error"] == "execution_admission_denied"
    assert invoker.results == []


@pytest.mark.asyncio
async def test_gate_b_missing_caller_identity() -> None:
    registry = ToolRegistry()
    registry.register(_contract("gate.b.ok"), _handler())
    invoker = _RecordingCatalogInvoker()
    mcp = _build_mcp(registry, invoker)
    task, run_id = await _register_binding()
    lab_identity = McpRunAgentIdentityResolution(
        tenant_id=_TENANT,
        user_id=_USER,
        principal=None,
        request_identity=None,
    )
    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
        return_value=lab_identity,
    ):
        async with Client(mcp) as client:
            with pytest.raises(ToolError, match="missing_caller_identity"):
                await client.call_tool(
                    "invoke_catalog_tool",
                    {"tool_id": "gate.b.ok", "task_id": str(task.task_id), "run_id": run_id},
                )


@pytest.mark.asyncio
async def test_gate_b_wrong_tenant() -> None:
    registry = ToolRegistry()
    registry.register(_contract("gate.b.ok"), _handler())
    invoker = _RecordingCatalogInvoker()
    mcp = _build_mcp(registry, invoker)
    task, run_id = await _register_binding(tenant=_TENANT)
    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
        return_value=_identity_resolution(tenant=_OTHER_TENANT),
    ):
        async with Client(mcp) as client:
            with pytest.raises(ToolError, match="wrong_tenant"):
                await client.call_tool(
                    "invoke_catalog_tool",
                    {"tool_id": "gate.b.ok", "task_id": str(task.task_id), "run_id": run_id},
                )


@pytest.mark.asyncio
async def test_gate_b_unknown_tool() -> None:
    registry = ToolRegistry()
    invoker = _RecordingCatalogInvoker()
    mcp = _build_mcp(registry, invoker)
    task, run_id = await _register_binding()
    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
        return_value=_identity_resolution(),
    ):
        async with Client(mcp) as client:
            with pytest.raises(ToolError, match="unknown_tool"):
                await client.call_tool(
                    "invoke_catalog_tool",
                    {"tool_id": "missing.tool", "task_id": str(task.task_id), "run_id": run_id},
                )


@pytest.mark.asyncio
async def test_gate_b_disabled_tool() -> None:
    registry = ToolRegistry()
    registry.register(_contract("gate.b.disabled"), _handler())
    invoker = _RecordingCatalogInvoker()
    mcp = _build_mcp(registry, invoker, enablement=_Enablement(set()))
    task, run_id = await _register_binding()
    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
        return_value=_identity_resolution(),
    ):
        async with Client(mcp) as client:
            with pytest.raises(ToolError, match="tool_disabled"):
                await client.call_tool(
                    "invoke_catalog_tool",
                    {
                        "tool_id": "gate.b.disabled",
                        "task_id": str(task.task_id),
                        "run_id": run_id,
                    },
                )


@pytest.mark.asyncio
async def test_gate_b_missing_execution_binding() -> None:
    registry = ToolRegistry()
    registry.register(_contract("gate.b.ok"), _handler())
    invoker = _RecordingCatalogInvoker()
    mcp = _build_mcp(registry, invoker)
    task_id = mint_task_id()
    run_id = mint_run_id()
    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
        return_value=_identity_resolution(),
    ):
        async with Client(mcp) as client:
            with pytest.raises(ToolError, match="missing_execution_binding"):
                await client.call_tool(
                    "invoke_catalog_tool",
                    {"tool_id": "gate.b.ok", "task_id": str(task_id), "run_id": str(run_id)},
                )


@pytest.mark.asyncio
async def test_gate_b_stale_run_binding() -> None:
    registry = ToolRegistry()
    registry.register(_contract("gate.b.ok"), _handler())
    invoker = _RecordingCatalogInvoker()
    mcp = _build_mcp(registry, invoker)
    task, _run_id = await _register_binding()
    stale_run = mint_run_id()
    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
        return_value=_identity_resolution(),
    ):
        async with Client(mcp) as client:
            with pytest.raises(ToolError, match="stale_run_binding"):
                await client.call_tool(
                    "invoke_catalog_tool",
                    {
                        "tool_id": "gate.b.ok",
                        "task_id": str(task.task_id),
                        "run_id": str(stale_run),
                    },
                )


@pytest.mark.asyncio
async def test_gate_b_unauthorized_side_effect_tool() -> None:
    registry = ToolRegistry()
    registry.register(_contract("gate.b.fx", side_effects=True), _handler())
    invoker = _RecordingCatalogInvoker()
    mcp = _build_mcp(registry, invoker)
    task, run_id = await _register_binding()
    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
        return_value=_identity_resolution(),
    ):
        async with Client(mcp) as client:
            result = await client.call_tool(
                "invoke_catalog_tool",
                {"tool_id": "gate.b.fx", "task_id": str(task.task_id), "run_id": run_id},
            )
    assert result.data["error"] == "execution_admission_denied"
    assert invoker.results == []


@pytest.mark.asyncio
async def test_gate_b_parent_cancel_before_invoke() -> None:
    registry = ToolRegistry()
    registry.register(_contract("gate.b.ok"), _handler())
    invoker = _RecordingCatalogInvoker()
    mcp = _build_mcp(registry, invoker)
    task, run_id = await _register_binding()
    CancellationCoordinator.request(task, reason="parent_cancel")
    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
        return_value=_identity_resolution(),
    ):
        async with Client(mcp) as client:
            result = await client.call_tool(
                "invoke_catalog_tool",
                {"tool_id": "gate.b.ok", "task_id": str(task.task_id), "run_id": run_id},
            )
    assert result.data["error"] == "execution_cancelled"
    assert invoker.results == []


@pytest.mark.asyncio
async def test_gate_b_tool_failure_before_effect() -> None:
    registry = ToolRegistry()
    registry.register(_contract("gate.b.ok"), _handler())
    invoker = _RecordingCatalogInvoker(fail_code="permission_error")
    mcp = _build_mcp(registry, invoker)
    task, run_id = await _register_binding()
    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
        return_value=_identity_resolution(),
    ):
        async with Client(mcp) as client:
            result = await client.call_tool(
                "invoke_catalog_tool",
                {"tool_id": "gate.b.ok", "task_id": str(task.task_id), "run_id": run_id},
            )
    assert result.data["error"] == "execution_admission_denied"
    assert invoker.results == []


@pytest.mark.asyncio
async def test_gate_b_unknown_effect_certainty() -> None:
    registry = ToolRegistry()
    registry.register(_contract("gate.b.ok"), _handler())
    invoker = _RecordingCatalogInvoker(uncertain=True)
    mcp = _build_mcp(registry, invoker)
    task, run_id = await _register_binding()
    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
        return_value=_identity_resolution(),
    ):
        async with Client(mcp) as client:
            result = await client.call_tool(
                "invoke_catalog_tool",
                {"tool_id": "gate.b.ok", "task_id": str(task.task_id), "run_id": run_id},
            )
    assert result.data["error"] == "execution_admission_denied"
    assert invoker.results == []


@pytest.mark.asyncio
async def test_gate_b_unauthorized_caller_same_tenant_different_user() -> None:
    registry = ToolRegistry()
    registry.register(_contract("gate.b.ok"), _handler())
    invoker = _RecordingCatalogInvoker()
    mcp = _build_mcp(registry, invoker)
    task, run_id = await _register_binding(user_id=_USER)
    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
        return_value=_identity_resolution(user_id=_OTHER_USER),
    ):
        async with Client(mcp) as client:
            with pytest.raises(ToolError, match="unauthorized_execution_caller"):
                await client.call_tool(
                    "invoke_catalog_tool",
                    {"tool_id": "gate.b.ok", "task_id": str(task.task_id), "run_id": run_id},
                )
    assert invoker.results == []


@pytest.mark.asyncio
async def test_gate_b_rejects_asserted_step_without_execution_binding() -> None:
    registry = ToolRegistry()
    registry.register(_contract("gate.b.ok"), _handler())
    invoker = _RecordingCatalogInvoker()
    mcp = _build_mcp(registry, invoker)
    task_id = mint_task_id()
    run_id = mint_run_id()
    task = Task(
        task_id=task_id,
        tenant_id=_TENANT,
        user_id=_USER,
        message="gate-b",
        context=TaskContext(capability="gate-b.cap"),
        metadata={},
        agent_id="agent-gate-b",
    )
    await ActiveTaskRegistry.register(task, run_id)
    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
        return_value=_identity_resolution(),
    ):
        async with Client(mcp) as client:
            with pytest.raises(ToolError, match="missing_execution_binding"):
                await client.call_tool(
                    "invoke_catalog_tool",
                    {
                        "tool_id": "gate.b.ok",
                        "task_id": str(task.task_id),
                        "run_id": str(run_id),
                        "step_id": "caller-only-step",
                    },
                )
    assert invoker.results == []


@pytest.mark.asyncio
async def test_gate_b_stale_step_binding() -> None:
    registry = ToolRegistry()
    registry.register(_contract("gate.b.ok"), _handler())
    invoker = _RecordingCatalogInvoker()
    mcp = _build_mcp(registry, invoker)
    task, run_id = await _register_binding(step_id=_STEP)
    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
        return_value=_identity_resolution(),
    ):
        async with Client(mcp) as client:
            with pytest.raises(ToolError, match="stale_step_binding"):
                await client.call_tool(
                    "invoke_catalog_tool",
                    {
                        "tool_id": "gate.b.ok",
                        "task_id": str(task.task_id),
                        "run_id": run_id,
                        "step_id": "other-step",
                    },
                )
    assert invoker.results == []


@pytest.mark.asyncio
async def test_gate_b_admission_refusal_never_starts_invoker() -> None:
    registry = ToolRegistry()
    registry.register(_contract("gate.b.ok"), _handler())
    invoker = _RecordingCatalogInvoker(delay_s=0.2)
    mcp = _build_mcp(registry, invoker)
    task, run_id = await _register_binding()

    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
        return_value=_identity_resolution(),
    ):
        async with Client(mcp) as client:
            result = await client.call_tool(
                "invoke_catalog_tool",
                {"tool_id": "gate.b.ok", "task_id": str(task.task_id), "run_id": run_id},
            )
    assert result.data["error"] == "execution_admission_denied"
    assert invoker.results == []
    assert not invoker.started.is_set()


@pytest.mark.asyncio
async def test_gate_b_duplicate_handoff_conflict_does_not_start_invoker() -> None:
    registry = ToolRegistry()
    registry.register(_contract("gate.b.ok"), _handler())
    invoker = _RecordingCatalogInvoker(delay_s=0.2)
    task, run_id = await _register_binding()
    supervision = build_host_root_execution_transport_detachment_supervisor()
    mcp = _build_mcp(
        registry,
        invoker,
        transport_detachment_supervision=supervision,
    )

    async def _block_forever() -> None:
        await asyncio.Event().wait()

    existing_launch = create_mcp_protected_host_execution_launch(_block_forever)
    admitted = await supervision.admit_root_execution_supervision(
        RootExecutionTransportAdmissionRequest(
            task_id=task.task_id,
            verified_tenant_id=_TENANT,
            caller_request_identity=_principal(),
            host_asyncio_task=existing_launch.host_task,
        )
    )
    assert admitted.disposition is RootExecutionTransportDetachmentHandoffDisposition.ACCEPTED

    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
        return_value=_identity_resolution(),
    ):
        async with Client(mcp) as client:
            result = await client.call_tool(
                "invoke_catalog_tool",
                {"tool_id": "gate.b.ok", "task_id": str(task.task_id), "run_id": run_id},
            )
    assert result.data["error"] == "execution_admission_denied"
    assert invoker.results == []
    assert not invoker.started.is_set()
    existing_launch.host_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await existing_launch.host_task


@pytest.mark.asyncio
async def test_gate_b_missing_mutation_boundary_denies_invoke() -> None:
    registry = ToolRegistry()
    registry.register(_contract("gate.b.ok"), _handler())
    invoker = _RecordingCatalogInvoker()
    mcp = _build_mcp(registry, invoker, with_boundary=False)
    task, run_id = await _register_binding()
    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
        return_value=_identity_resolution(),
    ):
        async with Client(mcp) as client:
            result = await client.call_tool(
                "invoke_catalog_tool",
                {"tool_id": "gate.b.ok", "task_id": str(task.task_id), "run_id": run_id},
            )
    assert result.data["error"] == "execution_admission_denied"
    assert invoker.results == []


@pytest.mark.asyncio
async def test_gate_b_missing_supervision_denies_invoke() -> None:
    registry = ToolRegistry()
    registry.register(_contract("gate.b.ok"), _handler())
    invoker = _RecordingCatalogInvoker()
    mcp = _build_mcp(registry, invoker, auto_wire_catalog_transport_supervision=False)
    task, run_id = await _register_binding()
    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
        return_value=_identity_resolution(),
    ):
        async with Client(mcp) as client:
            result = await client.call_tool(
                "invoke_catalog_tool",
                {"tool_id": "gate.b.ok", "task_id": str(task.task_id), "run_id": run_id},
            )
    assert result.data["error"] == "execution_admission_denied"
    assert invoker.results == []


@pytest.mark.asyncio
async def test_gate_b_side_effect_port_deny_blocks_invoke() -> None:
    class _DenyPort:
        def authorize(
            self,
            request: CollaborativeWorkEnforcementRequest,
            *,
            source_agent_id: str = "platform.orchestration.tool_invocation",
            source_step_id: str | None = None,
        ) -> MeaningfulSideEffectAuthorizationResult:
            _ = request, source_agent_id, source_step_id
            return MeaningfulSideEffectAuthorizationResult(
                permitted=False,
                decision=PolicyDecision(action=PolicyAction.DENY, reason="deny"),
                enforcement_result=MagicMock(),
                requires_governed_continuation=False,
            )

    registry = ToolRegistry()
    registry.register(_contract("gate.b.fx", side_effects=True), _handler())
    invoker = _RecordingCatalogInvoker()
    boundary_env = ApplicationEnvironmentProfile(application_profile=ApplicationProfile.PRODUCT)
    boundary = build_harness_control_plane_mutation_boundary(boundary_env)
    mcp = build_nexus_mcp_server(
        name="gate-b-mse-deny",
        host_execution=AsyncMock(spec=HostTaskExecutionPort),
        registry=AgentRegistry(),
        default_capability="gate-b.cap",
        default_tenant_id=_TENANT,
        default_user_id=_USER,
        tool_registry=registry,
        catalog_tool_invoker=invoker,
        task_control_mutation_boundary=boundary,
        meaningful_side_effect_authorization=_DenyPort(),
    )
    task, run_id = await _register_binding()
    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
        return_value=_identity_resolution(),
    ):
        async with Client(mcp) as client:
            result = await client.call_tool(
                "invoke_catalog_tool",
                {"tool_id": "gate.b.fx", "task_id": str(task.task_id), "run_id": run_id},
            )
    assert result.data["error"] == "execution_admission_denied"
    assert invoker.results == []


@pytest.mark.asyncio
async def test_gate_b_mismatched_invoker_agent_denied() -> None:
    registry = ToolRegistry()
    registry.register(_contract("gate.b.ok"), _handler())
    invoker = _RecordingCatalogInvoker(caller_agent_id="other-agent")
    mcp = _build_mcp(registry, invoker)
    task, run_id = await _register_binding(agent_id="agent-gate-b")
    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
        return_value=_identity_resolution(),
    ):
        async with Client(mcp) as client:
            with pytest.raises(ToolError, match="unauthorized_execution_caller"):
                await client.call_tool(
                    "invoke_catalog_tool",
                    {"tool_id": "gate.b.ok", "task_id": str(task.task_id), "run_id": run_id},
                )
    assert invoker.results == []


@pytest.mark.asyncio
async def test_gate_b_terminal_execution_binding_denied() -> None:
    registry = ToolRegistry()
    registry.register(_contract("gate.b.ok"), _handler())
    invoker = _RecordingCatalogInvoker()
    mcp = _build_mcp(registry, invoker)
    task_id = mint_task_id()
    run_id = mint_run_id()
    task = Task(
        task_id=task_id,
        tenant_id=_TENANT,
        user_id=_USER,
        message="gate-b",
        context=TaskContext(capability="gate-b.cap"),
        metadata={_METADATA_STEP_ID: _STEP},
        agent_id="agent-gate-b",
        state=TaskState.CANCELLED,
    )
    await ActiveTaskRegistry.register(task, run_id)
    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
        return_value=_identity_resolution(),
    ):
        async with Client(mcp) as client:
            with pytest.raises(ToolError, match="execution_terminal"):
                await client.call_tool(
                    "invoke_catalog_tool",
                    {"tool_id": "gate.b.ok", "task_id": str(task.task_id), "run_id": str(run_id)},
                )
    assert invoker.results == []


@pytest.mark.asyncio
async def test_cancel_c_parent_cancel_during_in_flight_invoke() -> None:
    registry = ToolRegistry()
    registry.register(_contract("gate.b.slow"), _handler())
    invoker = _RecordingCatalogInvoker(delay_s=0.4)
    mcp = _build_mcp(registry, invoker)
    task, run_id = await _register_binding()

    async def _cancel_mid_invoke() -> None:
        await asyncio.sleep(0.05)
        CancellationCoordinator.request(task, reason="governed_parent_cancel")

    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
        return_value=_identity_resolution(),
    ):
        async with Client(mcp) as client:
            cancel_task = asyncio.create_task(_cancel_mid_invoke())
            result = await client.call_tool(
                "invoke_catalog_tool",
                {"tool_id": "gate.b.slow", "task_id": str(task.task_id), "run_id": run_id},
            )
            await cancel_task
    assert not invoker.started.is_set()
    assert result.data["error"] == "execution_admission_denied"
    assert invoker.results == []


@pytest.mark.asyncio
async def test_cancel_c_late_success_cannot_override_cancelled_terminal() -> None:
    registry = ToolRegistry()
    registry.register(_contract("gate.b.ok"), _handler())
    invoker = _RecordingCatalogInvoker()
    mcp = _build_mcp(registry, invoker)
    task, run_id = await _register_binding()
    CancellationCoordinator.request(task, reason="terminal_parent_cancel")
    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
        return_value=_identity_resolution(),
    ):
        async with Client(mcp) as client:
            result = await client.call_tool(
                "invoke_catalog_tool",
                {"tool_id": "gate.b.ok", "task_id": str(task.task_id), "run_id": run_id},
            )
    assert result.data["error"] == "execution_cancelled"
    assert result.data.get("success") is not True
    assert invoker.results == []


@pytest.mark.asyncio
async def test_gate_b_no_second_runtime_tool_invoker_in_bridge() -> None:
    """Bridge must use injected invoker only (no shadow RuntimeToolInvoker construction)."""
    registry = ToolRegistry()
    registry.register(_contract("gate.b.ok"), _handler())
    invoker = _RecordingCatalogInvoker()
    composition = McpCatalogInvocationComposition(
        catalog_invoker=invoker,
        registry=registry,
    )
    assert composition.catalog_invoker is invoker
