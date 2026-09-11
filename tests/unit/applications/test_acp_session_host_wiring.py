# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

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
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.runtime_boundary_adapters import (
    application_profile_to_runtime_profile,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.agent_runtime_governance import AgentIdentity
from intergrax.runtime.agent_governance.authorization_boundary import (
    AgentRuntimeGovernanceBoundary,
)
from intergrax.runtime.agent_governance.capability_resolver import InMemoryCapabilityGrantResolver
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.applications._shared.wiring import build_application_registry
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.tools.registry import ToolProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

_TENANT = "tenant-a"


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


def _grant_tenant_from_declarative_invoker(
    invoker: CatalogDeclarativeToolInvoker,
    *,
    agent_id: str,
    tenant_id: str,
) -> str:
    rt_invoker = invoker.tool_invoker
    assert isinstance(rt_invoker, RuntimeToolInvoker)
    governance = rt_invoker._agent_runtime_governance  # noqa: SLF001
    assert isinstance(governance, AgentRuntimeGovernanceBoundary)
    resolver = governance._pipeline._capability_resolver  # noqa: SLF001
    assert isinstance(resolver, InMemoryCapabilityGrantResolver)
    grant = resolver.resolve_grant(AgentIdentity(agent_id=agent_id, tenant_id=tenant_id))
    assert grant is not None
    return grant.tenant_id


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


def test_build_acp_session_host_from_harness_strict_preserves_tenant_in_governance() -> None:
    manifest = _echo_manifest("acp_strict_tenant_preserve")
    lab_environment = ApplicationEnvironmentProfile.lab_defaults(
        profile_id="acp_strict_tenant_preserve.lab",
    )
    runtime = build_harness_host_runtime(
        manifest,
        lab_environment,
        tenant_id=_TENANT,
        use_in_memory_trace=True,
    )
    assert runtime.tenant_id == _TENANT
    runtime = replace(runtime, environment=_strict_lab_environment("acp_strict_tenant_preserve.lab"))

    host_ctx = build_acp_session_host_from_harness(runtime)
    invoker = host_ctx.declarative_tool_invoker
    assert isinstance(invoker, CatalogDeclarativeToolInvoker)
    assert invoker.production_mode is True
    assert _grant_tenant_from_declarative_invoker(
        invoker,
        agent_id="echo",
        tenant_id=_TENANT,
    ) == _TENANT


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
