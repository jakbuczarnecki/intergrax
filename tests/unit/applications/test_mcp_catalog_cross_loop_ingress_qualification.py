# © Artur Czarnecki. All rights reserved.

"""F3-R4 — cross-loop catalog child ingress (qualification configuration only)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from intergrax.applications._shared.mcp_harness_principal import McpRunAgentIdentityResolution
from pydantic import BaseModel

from intergrax.applications._shared.mcp_catalog_invocation_bridge import (
    McpCatalogInvocationComposition,
    _METADATA_STEP_ID,
    _serialize_tool_result,
    execute_mcp_catalog_tool_invocation,
)
from intergrax.applications._shared.production_catalog_tool_child_execution_wiring import (
    build_production_catalog_tool_child_execution_port,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.root_execution_child_work_ingress import (
    RootChildWorkIngressRefusal,
)
from intergrax.runtime.execution.catalog_tool_child_work_port import CatalogToolChildInvokeWork
from intergrax.runtime.execution.root_execution_child_work_ingress import (
    RootCatalogChildWorkTicket,
)
from intergrax.contracts.tool_execution_outcome import ToolExecutionResult
from intergrax.runtime.execution.host_root_catalog_child_ingress_draining_delegate import (
    HostRootCatalogChildIngressDrainingDelegate,
)
from intergrax.runtime.execution.root_execution_child_work_ingress import (
    RootExecutionChildWorkIngress,
)
from intergrax.runtime.governance.active_execution_authority import (
    bind_active_execution_authority,
    reset_active_execution_authority,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.tool_executor import ToolHandler

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-cross"
_USER = "operator-cross"
_STEP = "cross.catalog.step"


class _In(BaseModel):
    query: str = "x"


class _Out(BaseModel):
    value: str = "cross-ok"


@dataclass
class _RecordingCatalogInvoker:
    caller_agent_id: str = "agent-cross"
    results: list[ExecutionBoundCatalogToolInvokeRequest] = field(default_factory=list)

    def invoke(self, request: ExecutionBoundCatalogToolInvokeRequest) -> ToolExecutionResult[_Out]:
        self.results.append(request)
        return ToolExecutionResult.ok(_Out(value="cross-child"))


def _contract(tool_id: str) -> ToolContract:
    return ToolContract(
        tool_id=tool_id,
        name=tool_id,
        description="cross qual",
        input_schema=_In,
        output_schema=_Out,
        error_mapping={},
        side_effects=False,
        risk_level=ToolRiskLevel.LOW,
        category="cross",
    )


def _handler() -> ToolHandler[_In, _Out]:
    async def _run(_input: _In, _ctx: object) -> _Out:
        return _Out(value="unused")

    return _run


def _principal() -> RequestIdentity:
    return RequestIdentity(
        tenant_id=_TENANT,
        user_id=_USER,
        principal_type=PrincipalType.USER,
        auth_subject=_USER,
    )


@pytest.fixture(autouse=True)
async def _clear_tasks() -> None:
    ActiveTaskRegistry.clear_for_tests()
    yield
    ActiveTaskRegistry.clear_for_tests()


@pytest.mark.asyncio
async def test_cross_loop_ingress_while_root_delegate_active() -> None:
    """Decisive feasibility: HTTP-side submit while supervised root delegate is running."""
    registry = ToolRegistry()
    registry.register(_contract("cross.ok"), _handler())
    invoker = _RecordingCatalogInvoker()
    child_port = build_production_catalog_tool_child_execution_port()
    ingress = RootExecutionChildWorkIngress(
        catalog_child_execution=child_port,
        serialize_tool_result=_serialize_tool_result,
    )
    composition = McpCatalogInvocationComposition(
        catalog_invoker=invoker,
        registry=registry,
        catalog_tool_child_execution=child_port,
        root_child_work_ingress=ingress,
        enable_cross_loop_catalog_child=True,
    )

    task_id = mint_task_id()
    run_id = mint_run_id()
    task = Task(
        task_id=task_id,
        tenant_id=_TENANT,
        user_id=_USER,
        message="cross-root",
        context=TaskContext(capability="cross.cap"),
        metadata={_METADATA_STEP_ID: _STEP},
        agent_id="agent-cross",
    )
    await ActiveTaskRegistry.register(task, run_id)
    ingress.bind_supervised_task(task_id)

    async def _slow_root(_request: object) -> str:
        await asyncio.sleep(0.05)
        return "root-done"

    draining = HostRootCatalogChildIngressDrainingDelegate(
        _SlowDelegate(_slow_root),
        ingress=ingress,
    )

    parent_execution_id = mint_execution_id()
    authority_token = bind_active_execution_authority(
        ParentExecutionAuthority.unrestricted_root(),
    )
    identity_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=parent_execution_id,
        task_id=task_id,
    )
    governance_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id=_TENANT,
            workspace_id="ws-cross",
            principal_id=_USER,
        ),
    )
    task.metadata["workspace_id"] = "ws-cross"

    resolution = McpRunAgentIdentityResolution(
        tenant_id=_TENANT,
        user_id=_USER,
        principal=None,
        request_identity=_principal(),
    )

    async def _http_catalog_invoke() -> dict[str, object]:
        with patch(
            "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
            return_value=resolution,
        ):
            return await execute_mcp_catalog_tool_invocation(
                composition,
                tool_id="cross.ok",
                task_id=str(task_id),
                run_id=str(run_id),
                asserted_tenant_id=_TENANT,
            )

    http_task = asyncio.create_task(_http_catalog_invoke())
    root_result = await draining.execute(object())
    http_payload = await http_task

    reset_active_execution_governance_identity(governance_token)
    reset_active_execution_identity(identity_token)
    reset_active_execution_authority(authority_token)

    assert root_result == "root-done"
    assert http_payload.get("success") is True
    assert invoker.results
    assert str(invoker.results[0].task_id) == str(task_id)


class _SlowDelegate:
    def __init__(self, runner: object) -> None:
        self._runner = runner

    async def execute(self, request: object) -> object:
        run = getattr(self._runner, "__call__", None)
        if run is None:
            raise TypeError("runner must be callable")
        return await run(request)


def _minimal_work(task: Task, invoker: _RecordingCatalogInvoker) -> CatalogToolChildInvokeWork:
    from intergrax.contracts.execution_bound_catalog_tool_invocation import (
        ExecutionBoundCatalogToolInvokeRequest,
    )

    return CatalogToolChildInvokeWork(
        invoke_request=ExecutionBoundCatalogToolInvokeRequest(
            tool_id="cross.ok",
            input=_In(),
            tenant_id=_TENANT,
            task_id=str(task.task_id),
            run_id=str(task.task_id),
            agent_id="agent-cross",
            step_id=_STEP,
        ),
        governed_task=task,
        catalog_invoker=invoker,
        tool_id="cross.ok",
        step_id=_STEP,
        side_effects=False,
        caller=_principal(),
        resolved_agent_id="agent-cross",
    )


@pytest.mark.asyncio
async def test_option_c_default_denies_cross_loop_without_qual_flag() -> None:
    registry = ToolRegistry()
    registry.register(_contract("cross.ok"), _handler())
    invoker = _RecordingCatalogInvoker()
    child_port = build_production_catalog_tool_child_execution_port()
    ingress = RootExecutionChildWorkIngress(catalog_child_execution=child_port)
    composition = McpCatalogInvocationComposition(
        catalog_invoker=invoker,
        registry=registry,
        catalog_tool_child_execution=child_port,
        root_child_work_ingress=ingress,
        enable_cross_loop_catalog_child=False,
    )
    task_id = mint_task_id()
    run_id = mint_run_id()
    task = Task(
        task_id=task_id,
        tenant_id=_TENANT,
        user_id=_USER,
        message="deny",
        context=TaskContext(capability="cross.cap"),
        metadata={_METADATA_STEP_ID: _STEP},
        agent_id="agent-cross",
    )
    await ActiveTaskRegistry.register(task, run_id)
    resolution = McpRunAgentIdentityResolution(
        tenant_id=_TENANT,
        user_id=_USER,
        principal=None,
        request_identity=_principal(),
    )
    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
        return_value=resolution,
    ):
        outcome = await execute_mcp_catalog_tool_invocation(
            composition,
            tool_id="cross.ok",
            task_id=str(task_id),
            run_id=str(run_id),
            asserted_tenant_id=_TENANT,
        )
    assert outcome.get("error") == "execution_admission_denied"
    assert not invoker.results


@pytest.mark.asyncio
async def test_ingress_submit_rejected_when_task_not_bound() -> None:
    child_port = build_production_catalog_tool_child_execution_port()
    ingress = RootExecutionChildWorkIngress(catalog_child_execution=child_port, max_queue=2)
    task_id = mint_task_id()
    run_id = mint_run_id()
    task = Task(
        task_id=task_id,
        tenant_id=_TENANT,
        user_id=_USER,
        message="x",
        context=TaskContext(capability="c"),
        metadata={},
        agent_id="agent-cross",
    )
    invoker = _RecordingCatalogInvoker()
    ticket = RootCatalogChildWorkTicket(
        task_id=task_id,
        run_id=run_id,
        caller=_principal(),
        work=_minimal_work(task, invoker),
    )
    terminal = await ingress.submit_catalog_child_work(ticket, wait_timeout_s=0.2)
    assert terminal.refusal is RootChildWorkIngressRefusal.NOT_SUPERVISED
    assert terminal.success is False


@pytest.mark.asyncio
async def test_ingress_shutdown_refuses_pending_and_new() -> None:
    child_port = build_production_catalog_tool_child_execution_port()
    ingress = RootExecutionChildWorkIngress(catalog_child_execution=child_port, max_queue=4)
    task_id = mint_task_id()
    run_id = mint_run_id()
    task = Task(
        task_id=task_id,
        tenant_id=_TENANT,
        user_id=_USER,
        message="x",
        context=TaskContext(capability="c"),
        metadata={},
        agent_id="agent-cross",
    )
    invoker = _RecordingCatalogInvoker()
    ingress.bind_supervised_task(task_id)
    ticket = RootCatalogChildWorkTicket(
        task_id=task_id,
        run_id=run_id,
        caller=_principal(),
        work=_minimal_work(task, invoker),
    )
    submit_task = asyncio.create_task(
        ingress.submit_catalog_child_work(ticket, wait_timeout_s=5.0),
    )
    await asyncio.sleep(0.02)
    ingress.refuse_new_submissions()
    terminal = await submit_task
    assert terminal.refusal is RootChildWorkIngressRefusal.SHUTDOWN
    assert terminal.success is False


@pytest.mark.asyncio
async def test_ingress_queue_full_refuses_submit() -> None:
    child_port = build_production_catalog_tool_child_execution_port()
    ingress = RootExecutionChildWorkIngress(catalog_child_execution=child_port, max_queue=1)
    task_id = mint_task_id()
    run_id = mint_run_id()
    task = Task(
        task_id=task_id,
        tenant_id=_TENANT,
        user_id=_USER,
        message="x",
        context=TaskContext(capability="c"),
        metadata={},
        agent_id="agent-cross",
    )
    invoker = _RecordingCatalogInvoker()
    ingress.bind_supervised_task(task_id)
    ticket = RootCatalogChildWorkTicket(
        task_id=task_id,
        run_id=run_id,
        caller=_principal(),
        work=_minimal_work(task, invoker),
    )
    asyncio.create_task(ingress.submit_catalog_child_work(ticket, wait_timeout_s=30.0))
    await asyncio.sleep(0.01)
    terminal = await ingress.submit_catalog_child_work(ticket, wait_timeout_s=0.5)
    assert terminal.refusal is RootChildWorkIngressRefusal.QUEUE_FULL


@pytest.mark.asyncio
async def test_cross_loop_cancel_before_drain_returns_shutdown_or_denied() -> None:
    child_port = build_production_catalog_tool_child_execution_port()
    ingress = RootExecutionChildWorkIngress(catalog_child_execution=child_port, max_queue=4)
    task_id = mint_task_id()
    run_id = mint_run_id()
    task = Task(
        task_id=task_id,
        tenant_id=_TENANT,
        user_id=_USER,
        message="cancel",
        context=TaskContext(capability="c"),
        metadata={_METADATA_STEP_ID: _STEP},
        agent_id="agent-cross",
    )
    invoker = _RecordingCatalogInvoker()
    ingress.bind_supervised_task(task_id)
    ticket = RootCatalogChildWorkTicket(
        task_id=task_id,
        run_id=run_id,
        caller=_principal(),
        work=_minimal_work(task, invoker),
    )
    pending = asyncio.create_task(
        ingress.submit_catalog_child_work(ticket, wait_timeout_s=5.0),
    )
    await asyncio.sleep(0.02)
    ingress.refuse_new_submissions()
    terminal = await pending
    assert terminal.success is False
    assert terminal.refusal in {
        RootChildWorkIngressRefusal.SHUTDOWN,
        RootChildWorkIngressRefusal.SUBMIT_TIMEOUT,
    }
