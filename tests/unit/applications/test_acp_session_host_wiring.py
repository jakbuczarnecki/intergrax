# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from contextvars import Token
from dataclasses import dataclass, replace
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from echo.echo_agent import EchoAgent
from intergrax.agents.persistence.catalog_declarative_invoker import (
    CatalogDeclarativeToolInvoker,
)
from intergrax.applications._shared.acp_session_host_wiring import (
    build_acp_session_host_from_harness,
)
from intergrax.applications._shared.agent_runtime_governance_wiring import (
    AgentRuntimeGovernanceMaterializationError,
)
from intergrax.applications._shared.harness_host_composition import (
    HarnessHostInternalComposition,
    HarnessHostPluginRegistrationSurface,
)
from intergrax.applications._shared.harness_host_runtime import (
    HarnessHostRuntime,
    build_harness_host_runtime,
)
from intergrax.applications._shared.runtime_boundary_adapters import (
    application_profile_to_runtime_profile,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    bind_active_execution_identity,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.applications._shared.wiring import build_application_registry
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.tools.core.contracts import ToolRiskLevel
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolProfile
from intergrax.tools.tool_executor import ToolHandler
from testing_support.builder import tools_agent_make_contract

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

_TENANT = "tenant-a"
_OTHER_TENANT = "tenant-b"
_PROBE_TOOL_ID = "acp.tenant_governance_probe"
_ATTEMPT_ID = AttemptId("attempt_01234567890123456789012345678901")
_EXECUTION_ID = ExecutionId("exec_01234567890123456789012345678901")


class _ProbeInput(BaseModel):
    pass


class _ProbeOutput(BaseModel):
    ok: bool = True


@dataclass
class _ExecutionProbe:
    count: int = 0


class _ProbeHandler(ToolHandler[BaseModel, BaseModel]):
    def __init__(self, probe: _ExecutionProbe) -> None:
        self._probe = probe

    def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
        _ = request
        self._probe.count += 1
        return _ProbeOutput()


def _echo_manifest(app_id: str) -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id=app_id,
        name="ACP Session Host Tenant",
        route_prefix=f"/v1/{app_id}",
        env_prefix=f"{app_id.upper()}_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )


def _strict_lab_environment(profile_id: str) -> ApplicationEnvironmentProfile:
    profile = ApplicationEnvironmentProfile.lab_defaults(profile_id=profile_id)
    return profile.model_copy(
        update={"meta": profile.meta.model_copy(update={"execution_mode": ExecutionMode.STRICT})},
    )


def _register_tenant_probe_tool(runtime: HarnessHostRuntime, probe: _ExecutionProbe) -> None:
    registry = runtime.env_wiring.tool_wiring.registry
    contract = replace(
        tools_agent_make_contract(_PROBE_TOOL_ID, _ProbeInput, _ProbeOutput),
        category="echo.basic",
        risk_level=ToolRiskLevel.LOW,
    )
    registry.register(contract, _ProbeHandler(probe))


def _strict_harness_with_probe(
    app_id: str,
    *,
    tenant_id: str,
) -> tuple[HarnessHostRuntime, _ExecutionProbe]:
    manifest = _echo_manifest(app_id)
    lab_environment = ApplicationEnvironmentProfile.lab_defaults(profile_id=f"{app_id}.lab")
    runtime = build_harness_host_runtime(
        manifest,
        lab_environment,
        tenant_id=tenant_id,
        use_in_memory_trace=True,
    )
    runtime = replace(runtime, environment=_strict_lab_environment(f"{app_id}.lab"))
    probe = _ExecutionProbe()
    _register_tenant_probe_tool(runtime, probe)
    return runtime, probe


def _bind_declarative_invoke_governance(*, run_id: str) -> tuple[Token[object], Token[object]]:
    identity_token = bind_active_execution_identity(
        run_id=RunId(run_id),
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    )
    budget_token = bind_root_execution_budget(
        execution_id=_EXECUTION_ID,
        ledger=create_execution_budget_ledger(RunBudget()),
    )
    return identity_token, budget_token


def test_build_acp_session_host_from_harness_attaches_decision_gate() -> None:
    decision_gate = MagicMock()
    runtime = MagicMock()
    runtime.tenant_id = ""
    runtime.environment = ApplicationEnvironmentProfile.lab_defaults()
    nexus_loop = MagicMock()
    nexus_loop.peek_decision_flow_gate.return_value = decision_gate
    runtime._internal_composition = HarnessHostInternalComposition(
        execution_terminal=MagicMock(),
        event_bus=MagicMock(),
        decision_flow_gate=decision_gate,
        middleware_pipeline=MagicMock(),
        lifecycle_hook_coordinator=MagicMock(),
        plugin_surface=HarnessHostPluginRegistrationSurface(
            event_bus=MagicMock(),
            hook_registry=MagicMock(),
            policy_engine=MagicMock(),
        ),
        runtime_event_persistence=None,
        _orchestration_backend=nexus_loop,
    )
    tool_wiring = MagicMock()
    tool_wiring.profile.enabled = False
    tool_wiring.profile.enabled_bundles = []
    runtime.env_wiring.tool_wiring = tool_wiring
    runtime.manifest = MagicMock()
    runtime.registry = MagicMock()
    runtime.reliability = MagicMock(idempotency_store=None)

    host_ctx = build_acp_session_host_from_harness(runtime)
    assert host_ctx.decision_flow_gate is decision_gate
    assert host_ctx.runtime_profile == application_profile_to_runtime_profile(runtime.environment)


@pytest.mark.asyncio
async def test_build_acp_session_host_from_harness_strict_tenant_governance_allows_matching_tenant() -> (
    None
):
    runtime, probe = _strict_harness_with_probe("acp_strict_tenant_preserve", tenant_id=_TENANT)
    assert runtime.tenant_id == _TENANT

    host_ctx = build_acp_session_host_from_harness(runtime)
    invoker = host_ctx.declarative_tool_invoker
    assert isinstance(invoker, CatalogDeclarativeToolInvoker)
    assert invoker.production_mode is True

    run_id = mint_run_id()
    task_id = mint_task_id()
    invoker.bind_run(
        run_id=run_id,
        task_id=task_id,
        agent_id="echo",
        tenant_id=_TENANT,
    )
    identity_token, budget_token = _bind_declarative_invoke_governance(run_id=run_id)
    try:
        result = await invoker.invoke(
            tool_id=_PROBE_TOOL_ID,
            args={},
            idempotency_key="acp-tenant-positive",
        )
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_identity(identity_token)

    assert result.status == "success"
    assert probe.count == 1


@pytest.mark.asyncio
async def test_build_acp_session_host_from_harness_strict_tenant_governance_denies_wrong_tenant() -> (
    None
):
    runtime, probe = _strict_harness_with_probe("acp_strict_tenant_isolate", tenant_id=_TENANT)
    host_ctx = build_acp_session_host_from_harness(runtime)
    invoker = host_ctx.declarative_tool_invoker
    assert isinstance(invoker, CatalogDeclarativeToolInvoker)

    run_id = mint_run_id()
    task_id = mint_task_id()
    invoker.bind_run(
        run_id=run_id,
        task_id=task_id,
        agent_id="echo",
        tenant_id=_OTHER_TENANT,
    )
    identity_token, budget_token = _bind_declarative_invoke_governance(run_id=run_id)
    try:
        result = await invoker.invoke(
            tool_id=_PROBE_TOOL_ID,
            args={},
            idempotency_key="acp-tenant-negative",
        )
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_identity(identity_token)

    assert result.status != "success"
    assert probe.count == 0


def test_build_acp_session_host_from_harness_strict_missing_tenant_fails_closed() -> None:
    manifest = _echo_manifest("acp_strict_missing_tenant")
    build_ctx = ApplicationBuildContext.for_manifest(manifest, policy_bundle=RuntimePolicyBundle())
    registry = build_application_registry(manifest, build_ctx)
    runtime = MagicMock()
    runtime.tenant_id = "   "
    runtime.environment = _strict_lab_environment("acp_strict_missing_tenant.lab")
    runtime.env_wiring = MagicMock()
    tool_wiring = MagicMock()
    tool_wiring.profile = ToolProfile(enabled=["read_file"])
    tool_wiring.registry = MagicMock()
    tool_wiring.wiring_context = MagicMock()
    runtime.env_wiring.tool_wiring = tool_wiring
    runtime.manifest = manifest
    runtime.registry = registry
    runtime.reliability = MagicMock(idempotency_store=None)
    runtime._internal_composition = HarnessHostInternalComposition(
        execution_terminal=MagicMock(),
        event_bus=MagicMock(),
        decision_flow_gate=MagicMock(),
        middleware_pipeline=MagicMock(),
        lifecycle_hook_coordinator=MagicMock(),
        plugin_surface=HarnessHostPluginRegistrationSurface(
            event_bus=MagicMock(),
            hook_registry=MagicMock(),
            policy_engine=MagicMock(),
        ),
        runtime_event_persistence=None,
        _orchestration_backend=MagicMock(),
    )

    with pytest.raises(AgentRuntimeGovernanceMaterializationError, match="tenant_id"):
        build_acp_session_host_from_harness(runtime)
