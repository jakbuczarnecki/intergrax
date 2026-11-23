# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from contextvars import Token
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
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
from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime
from intergrax.applications._shared.runtime_boundary_adapters import (
    application_profile_to_runtime_profile,
)
from intergrax.applications._shared.wiring import build_application_registry
from intergrax.applications.contracts.build_context import ApplicationBuildContext
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
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.shipped_manifests import INMEMORY
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.nexus.agents.catalog_declarative_invoker import (
    CatalogDeclarativeToolInvoker,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.skills.registry.profile import SkillProfile
from intergrax.tools.core.contracts import ToolRiskLevel
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolProfile
from intergrax.tools.tool_executor import ToolHandler
from pydantic import BaseModel
from testing_support.agent_platform_admin_harness import (
    allow_mutation_boundary,
    lifecycle_proof_durable_profile_stores,
)
from testing_support.application_harness_test_support import build_harness_host_runtime_for_tests
from testing_support.builder import tools_agent_make_contract
from testing_support.dependency_concurrency_admission_config import (
    tool_dependency_concurrency_admission_configuration,
)
from tests.unit.applications.ac3_projection_helpers import build_test_registry_projection

from echo.echo_agent import EchoAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

_TENANT = "tenant-a"
_OTHER_TENANT = "tenant-b"
_PROBE_TOOL_ID = "acp.tenant_governance_probe"
_ATTEMPT_ID = AttemptId("attempt_01234567890123456789012345678901")
_EXECUTION_ID = ExecutionId("exec_01234567890123456789012345678901")


class _HarnessDurableKV(DistributedKVStore):
    """Minimal durable-marked KV for STRICT harness composition (skill execution pinning)."""

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], bytes] = {}

    @property
    def survives_process_restart(self) -> bool:
        return True

    def get(self, tenant_id: str, key: str) -> bytes | None:
        return self._data.get((tenant_id, key))

    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        _ = ttl_seconds
        self._data[(tenant_id, key)] = value

    def delete(self, tenant_id: str, key: str) -> None:
        self._data.pop((tenant_id, key), None)

    def compare_and_set(
        self,
        tenant_id: str,
        key: str,
        expected: bytes | None,
        new_value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> bool:
        _ = ttl_seconds
        current = self._data.get((tenant_id, key))
        if current != expected:
            return False
        self._data[(tenant_id, key)] = new_value
        return True


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
    strict = profile.model_copy(
        update={"meta": profile.meta.model_copy(update={"execution_mode": ExecutionMode.STRICT})},
    )
    if strict.reliability_profile.dependency_concurrency_admission is None:
        strict = strict.model_copy(
            update={
                "reliability_profile": strict.reliability_profile.model_copy(
                    update={
                        "dependency_concurrency_admission": tool_dependency_concurrency_admission_configuration(
                            _PROBE_TOOL_ID,
                        ),
                    },
                ),
            },
        )
    integration = strict.integration_profile.model_copy(update={"vector_store": INMEMORY})
    memory = strict.memory_profile.model_copy(
        update={
            "enable_long_term_memory": False,
            "enable_session_vector_index": False,
        },
    )
    return strict.model_copy(
        update={
            "capabilities": strict.capabilities.model_copy(update={"integrations": integration}),
            "context_profile": strict.context_profile.model_copy(
                update={"enable_rag": False, "enable_websearch": False},
            ),
            "memory_profile": memory,
        },
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
    tmp_path: Path,
) -> tuple[HarnessHostRuntime, _ExecutionProbe]:
    manifest = _echo_manifest(app_id)
    strict_environment = _strict_lab_environment(f"{app_id}.lab")
    registry_projection = build_test_registry_projection(
        manifest,
        strict_environment,
        builders={EchoAgent: lambda _ctx, _binding: EchoAgent()},
    )
    profile_stores = lifecycle_proof_durable_profile_stores(tmp_path)
    runtime = build_harness_host_runtime_for_tests(
        manifest,
        strict_environment,
        tenant_id=tenant_id,
        use_in_memory_trace=False,
        trace_db_path=tmp_path / "acp-trace.db",
        runtime_events_db_path=tmp_path / "acp-runtime-events.db",
        checkpoints_db_path=tmp_path / "acp-checkpoints.db",
        idempotency_db_path=tmp_path / "acp-idempotency.db",
        document_store=InMemoryDocumentStore(),
        key_value_cache=_HarnessDurableKV(),
        registry_projection=registry_projection,
        mutation_authorization_boundary=allow_mutation_boundary(),
        revision_store=profile_stores.revision_store,
        pinning_store=profile_stores.pinning_store,
        active_store=profile_stores.active_store,
    )
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
        ),
        runtime_event_persistence=None,
        plugin_bootstrap_target=nexus_loop,
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
async def test_build_acp_session_host_from_harness_strict_tenant_governance_allows_matching_tenant(
    tmp_path: Path,
) -> None:
    runtime, probe = _strict_harness_with_probe(
        "acp_strict_tenant_preserve",
        tenant_id=_TENANT,
        tmp_path=tmp_path,
    )
    assert runtime.tenant_id == _TENANT

    host_ctx = build_acp_session_host_from_harness(runtime)
    invoker = host_ctx.declarative_tool_invoker
    assert isinstance(invoker, CatalogDeclarativeToolInvoker)
    assert invoker.production_mode is True

    run_id = mint_run_id()
    task_id = mint_task_id()
    identity_token, budget_token = _bind_declarative_invoke_governance(run_id=run_id)
    try:
        result = await invoker.invoke(
            tenant_id=_TENANT,
            run_id=str(run_id),
            task_id=str(task_id),
            agent_id="echo",
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
async def test_build_acp_session_host_from_harness_strict_tenant_governance_denies_wrong_tenant(
    tmp_path: Path,
) -> None:
    runtime, probe = _strict_harness_with_probe(
        "acp_strict_tenant_isolate",
        tenant_id=_TENANT,
        tmp_path=tmp_path,
    )
    host_ctx = build_acp_session_host_from_harness(runtime)
    invoker = host_ctx.declarative_tool_invoker
    assert isinstance(invoker, CatalogDeclarativeToolInvoker)

    run_id = mint_run_id()
    task_id = mint_task_id()
    identity_token, budget_token = _bind_declarative_invoke_governance(run_id=run_id)
    try:
        result = await invoker.invoke(
            tenant_id=_OTHER_TENANT,
            run_id=str(run_id),
            task_id=str(task_id),
            agent_id="echo",
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
    from intergrax.applications._shared.application_composition_context import (
        composition_for_factory_context,
    )

    build_ctx = ApplicationBuildContext.for_manifest(manifest)
    composition = composition_for_factory_context(
        build_ctx,
        policy_bundle=RuntimePolicyBundle(),
        skill_profile=SkillProfile(enabled_bundles=["harness"]),
    )
    registry = build_application_registry(manifest, build_ctx, composition=composition)
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
        ),
        runtime_event_persistence=None,
        plugin_bootstrap_target=MagicMock(),
    )

    with pytest.raises(AgentRuntimeGovernanceMaterializationError, match="tenant_id"):
        build_acp_session_host_from_harness(runtime)
