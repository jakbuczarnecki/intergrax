# © Artur Czarnecki. All rights reserved.

"""TR-01-RQ-C1C — capability evidence completeness and authority/governance separation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.applications.contracts.environment_profile.bundles import IsolationBundle
from intergrax.applications.contracts.environment_profile.sub_profiles import SandboxProfile
from intergrax.contracts.runtime_sandbox_isolation_authority import RuntimeSandboxIsolationAuthority
from intergrax.contracts.sandbox_profile import SandboxProfile as ContractSandboxProfile
from intergrax.runtime.nexus.errors.declarative_policy_violation_error import DeclarativePolicyViolationError
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.sandbox.contracts import (
    SandboxExecCapable,
    SandboxSecurityCapabilities,
    SandboxSecurityCapable,
)
from intergrax.runtime.sandbox.enforcement import resolve_tool_execution_environment
from intergrax.runtime.sandbox.execution_environment import (
    ExecutionEnvironmentProviderKind,
    FilesystemAccess,
    ProcessExecution,
)
from intergrax.runtime.sandbox.isolation_gate import sandbox_availability_provider
from intergrax.runtime.sandbox.provider_adapters import (
    capabilities_from_attested_exec_session,
    capabilities_from_host_backend,
    probe_provider_capabilities_from_wiring,
    project_provider_capabilities_from_security,
)
from intergrax.runtime.sandbox.resolver import resolve_effective_execution_environment_for_profile
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.runtime.sandbox.execution_environment import ExecutionEnvironmentRequirement
from intergrax.tools.core.contracts import ToolIsolationRequirement
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.providers.sandbox.bundle import register_sandbox_tools, sandbox_exec_contract
from intergrax.tools.providers.sandbox.contracts import SandboxExecInput
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from testing_support.builder import build_runtime_state_for_tests, canonical_governed_execution_scope

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _PlainExecOnly:
    session_id = "plain-exec"

    def execute(self, operation: str, payload: dict | None = None) -> object:
        return MagicMock(output={}, session_id=self.session_id)


class _PartialAttestedProvider:
    session_id = "partial"

    def execute(self, operation: str, payload: dict | None = None) -> object:
        return MagicMock(output={}, session_id=self.session_id)

    def security_capabilities(self) -> SandboxSecurityCapabilities:
        return SandboxSecurityCapabilities(
            isolation_tier="local",
            provider_id="partial:1",
            supports_sandboxed_exec=None,
            supports_workspace_write=None,
            filesystem_access=None,
            process_execution=None,
        )


class _FullAttestedProvider:
    session_id = "full"

    def execute(self, operation: str, payload: dict | None = None) -> object:
        return MagicMock(output={}, session_id=self.session_id)

    def security_capabilities(self) -> SandboxSecurityCapabilities:
        return SandboxSecurityCapabilities(
            isolation_tier="container",
            provider_id="plugin:full",
            network_egress_deny_enforced=True,
            supports_sandboxed_exec=True,
            supports_workspace_write=True,
            filesystem_access=FilesystemAccess.WORKSPACE_WRITE,
            process_execution=ProcessExecution.SANDBOXED,
        )


def _explicit_authority() -> RuntimeSandboxIsolationAuthority:
    return RuntimeSandboxIsolationAuthority(
        sandbox=ContractSandboxProfile(enable_exec_tool=True),
    )


def _deny_policy_bundle() -> object:
    profile = ApplicationEnvironmentProfile.lab_defaults(profile_id="c1c-deny")
    profile = profile.model_copy(
        update={
            "isolation": IsolationBundle(sandbox=SandboxProfile(enable_exec_tool=True)),
        },
    )
    profile.policy_rules = PolicyRulesProfile(
        inline_rules=[
            {
                "rule_id": "c1c-deny-sandbox-exec",
                "handler_id": "deny_tool",
                "resource_kind": "tool",
                "resource_id": "sandbox.exec",
                "action": "deny",
            },
        ],
        policy_enforcement_mode="enforce",
    )
    return wire_policy_bundle(profile)


@pytest.fixture
def sandbox_session(tmp_path: Path) -> SandboxSession:
    return SandboxSession.create(tmp_path, tenant_id="tenant-1", task_id="task-1")


def test_c1c_b1_plain_exec_no_security_capabilities() -> None:
    assert capabilities_from_attested_exec_session(_PlainExecOnly()) is None


def test_c1c_b6_unknown_isolation_tier_fails_closed() -> None:
    security = SandboxSecurityCapabilities(
        isolation_tier="invalid-tier",  # type: ignore[arg-type]
        provider_id="x",
        supports_sandboxed_exec=True,
        supports_workspace_write=True,
        filesystem_access=FilesystemAccess.WORKSPACE_WRITE,
        process_execution=ProcessExecution.SANDBOXED,
    )
    assert project_provider_capabilities_from_security(security) is None


def test_c1c_b7_attested_local_provider_projection(sandbox_session: SandboxSession) -> None:
    caps = project_provider_capabilities_from_security(sandbox_session.security_capabilities())
    assert caps is not None
    assert caps.provider_ref.provider_kind is ExecutionEnvironmentProviderKind.LOCAL


def test_c1c_b8_attested_container_maps_to_hosted() -> None:
    caps = project_provider_capabilities_from_security(_FullAttestedProvider().security_capabilities())
    assert caps is not None
    assert caps.provider_ref.provider_kind is ExecutionEnvironmentProviderKind.HOSTED


def test_c1c_b9_custom_fully_attested_provider_resolves() -> None:
    ctx = ToolWiringContext(
        sandbox_session=_FullAttestedProvider(),
        sandbox_isolation_authority=_explicit_authority(),
    )
    _, err = resolve_tool_execution_environment(ctx, contract=sandbox_exec_contract())
    assert err is None


def test_c1c_b10_partial_attestation_fails_closed() -> None:
    ctx = ToolWiringContext(
        sandbox_session=_PartialAttestedProvider(),
        sandbox_isolation_authority=_explicit_authority(),
    )
    assert probe_provider_capabilities_from_wiring(ctx) == ()
    _, err = resolve_tool_execution_environment(ctx, contract=sandbox_exec_contract())
    assert err is not None


def test_c1c_h1_unattested_host_backend_no_projection() -> None:
    backend = MagicMock()
    assert capabilities_from_host_backend(backend) is None


def test_c1c_h4_isolation_tool_unattested_host_zero_providers() -> None:
    backend = MagicMock()
    ctx = ToolWiringContext(sandbox_host=backend, sandbox_isolation_authority=_explicit_authority())
    assert probe_provider_capabilities_from_wiring(ctx) == ()


def test_c1c_r3_authority_without_provider_capability_fails(sandbox_session: SandboxSession) -> None:
    ctx = ToolWiringContext(
        sandbox_session=_PlainExecOnly(),
        sandbox_isolation_authority=_explicit_authority(),
    )
    _, err = resolve_tool_execution_environment(ctx, contract=sandbox_exec_contract())
    assert err is not None


def test_c1c_r5_authority_complete_capability_governance_allow(sandbox_session: SandboxSession) -> None:
    ctx = ToolWiringContext(
        sandbox_session=sandbox_session,
        sandbox_isolation_authority=_explicit_authority(),
    )
    env, err = resolve_tool_execution_environment(ctx, contract=sandbox_exec_contract())
    assert err is None
    assert env is not None


def test_c1c_q_governance_deny_zero_physical_execute(sandbox_session: SandboxSession) -> None:
    ctx = ToolWiringContext(
        sandbox_session=sandbox_session,
        sandbox_isolation_authority=_explicit_authority(),
    )
    registry = ToolRegistry()
    register_sandbox_tools(registry, ctx)
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
        sandbox_availability=sandbox_availability_provider(ctx),
    )
    state = build_runtime_state_for_tests(run_id="c1c_gov")
    state = replace(
        state,
        context=replace(state.context, config=replace(state.context.config, policy_bundle=_deny_policy_bundle())),
    )
    request = ToolExecutionRequest(
        run_id="c1c_gov",
        step_id="step/1",
        tool_id="sandbox.exec",
        input=SandboxExecInput(operation="echo", payload={"message": "blocked"}),
    )
    audit_before = len(sandbox_session.audit_log)
    with canonical_governed_execution_scope("c1c_gov"):
        with pytest.raises(DeclarativePolicyViolationError):
            invoker.invoke(state=state, agent_id="agent", request=request)
    assert len(sandbox_session.audit_log) == audit_before


def test_c1c_b5_missing_provider_id_fails_projection() -> None:
    security = SandboxSecurityCapabilities(
        isolation_tier="local",
        provider_id="   ",
        supports_sandboxed_exec=True,
        supports_workspace_write=True,
        filesystem_access=FilesystemAccess.WORKSPACE_WRITE,
        process_execution=ProcessExecution.SANDBOXED,
    )
    assert project_provider_capabilities_from_security(security) is None


def test_c1c_b2_missing_sandboxed_process_evidence_fails_resolution() -> None:
    security = SandboxSecurityCapabilities(
        isolation_tier="local",
        provider_id="weak-process",
        supports_sandboxed_exec=False,
        supports_workspace_write=True,
        filesystem_access=FilesystemAccess.WORKSPACE_WRITE,
        process_execution=ProcessExecution.DENIED,
    )
    caps = project_provider_capabilities_from_security(security)
    assert caps is not None
    result = resolve_effective_execution_environment_for_profile(
        _explicit_authority(),
        ExecutionEnvironmentRequirement.from_tool_isolation(ToolIsolationRequirement.SANDBOX),
        caps,
    )
    assert result.failure is not None
