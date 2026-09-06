# © Artur Czarnecki. All rights reserved.

"""P1.8 — effective execution environment convergence proofs."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from intergrax.applications._shared.profile_resolution import (
    InMemoryEffectiveProfileExecutionPinningStore,
    InMemoryEffectiveProfileRevisionStore,
    materialize_effective_profile_revision,
    pin_effective_profile_revision_for_execution,
    resolve_profile,
)
from intergrax.applications._shared.runtime_inspection.service import RuntimeInspectionService
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.bundles import IsolationBundle
from intergrax.applications.contracts.environment_profile.sub_profiles import SandboxProfile
from intergrax.applications.contracts.profile_resolution import EffectiveProfileRevisionScope
from intergrax.integrations.contracts.sandbox_host import SandboxExecResult, SandboxSession as HostSession
from intergrax.runtime.sandbox import enforcement as enforcement_module
from intergrax.runtime.sandbox.enforcement import resolve_tool_execution_environment
from intergrax.runtime.sandbox.execution_environment import (
    ExecutionEnvironmentProviderKind,
    ExecutionEnvironmentProviderRef,
    ExecutionEnvironmentRequirement,
    ExecutionEnvironmentResolutionFailureReason,
    FilesystemAccess,
    NetworkAccess,
    ProcessExecution,
    PrivilegeMode,
    SandboxProviderCapabilities,
)
from intergrax.runtime.sandbox.hosted_session import HostedSandboxSession
from intergrax.runtime.sandbox.provider_adapters import (
    capabilities_from_host_backend,
    capabilities_from_local_session,
    probe_provider_capabilities_from_wiring,
)
from intergrax.runtime.sandbox.resolver import (
    profile_isolation_authority,
    resolve_effective_execution_environment,
    resolve_effective_execution_environment_for_profile,
    resolve_effective_execution_environment_for_revision,
    validate_child_requirement_not_widening,
)
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.tools.core.contracts import ToolIsolationRequirement
from intergrax.tools.providers.sandbox.bundle import sandbox_exec_contract
from intergrax.tools.providers.sandbox.contracts import SandboxExecInput
from intergrax.tools.providers.sandbox.service import sandbox_exec
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.contracts.execution_identity import ExecutionId, mint_execution_id

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _sandbox_profile(*, enable_exec_tool: bool = True) -> ApplicationEnvironmentProfile:
    profile = ApplicationEnvironmentProfile.lab_defaults(profile_id="p1-8")
    return profile.model_copy(
        update={
            "isolation": IsolationBundle(
                sandbox=SandboxProfile(enable_exec_tool=enable_exec_tool),
            ),
        },
    )


def _no_sandbox_profile() -> ApplicationEnvironmentProfile:
    profile = ApplicationEnvironmentProfile.lab_defaults(profile_id="p1-8-none")
    return profile.model_copy(update={"isolation": IsolationBundle()})


def _local_provider(session: SandboxSession) -> SandboxProviderCapabilities:
    return capabilities_from_local_session(session)


def _hosted_provider(*, supports_exec: bool = True) -> SandboxProviderCapabilities:
    return SandboxProviderCapabilities(
        provider_ref=ExecutionEnvironmentProviderRef(
            provider_id="hosted:test",
            provider_kind=ExecutionEnvironmentProviderKind.HOSTED,
        ),
        filesystem_access=FilesystemAccess.WORKSPACE_WRITE,
        network_access=NetworkAccess.RESTRICTED,
        process_execution=ProcessExecution.SANDBOXED if supports_exec else ProcessExecution.DENIED,
        supports_sandboxed_exec=supports_exec,
        supports_workspace_write=True,
        supports_network_isolation=None,
    )


def _requirement_sandbox() -> ExecutionEnvironmentRequirement:
    return ExecutionEnvironmentRequirement.from_tool_isolation(ToolIsolationRequirement.SANDBOX)


@pytest.fixture
def sandbox_session(tmp_path: Path) -> SandboxSession:
    return SandboxSession.create(
        tmp_path,
        tenant_id="tenant-a",
        task_id="task-a",
        allowed_operations=frozenset(
            {"echo", "write_file", "read_file", "list_files", "run_python"},
        ),
    )


def test_profile_without_sandbox_is_conservative_authority() -> None:
    authority = profile_isolation_authority(_no_sandbox_profile())
    assert authority.sandbox_configured is False
    assert authority.filesystem_access is FilesystemAccess.NONE
    assert authority.process_execution is ProcessExecution.DENIED


def test_profile_with_sandbox_reflects_configured_capability() -> None:
    authority = profile_isolation_authority(_sandbox_profile())
    assert authority.sandbox_configured is True
    assert authority.filesystem_access is FilesystemAccess.WORKSPACE_WRITE
    assert authority.network_access is NetworkAccess.NONE
    assert authority.process_execution is ProcessExecution.SANDBOXED
    assert authority.privilege_mode is PrivilegeMode.STANDARD


def test_requirement_success_when_profile_provider_and_requirement_align(
    sandbox_session: SandboxSession,
) -> None:
    result = resolve_effective_execution_environment_for_profile(
        _sandbox_profile(),
        _requirement_sandbox(),
        _local_provider(sandbox_session),
    )
    assert result.failure is None
    assert result.environment is not None
    assert result.environment.sandbox_required is True
    assert result.environment.process_execution is ProcessExecution.SANDBOXED
    assert result.environment.provider_ref.provider_kind is ExecutionEnvironmentProviderKind.LOCAL


def test_authority_violation_when_profile_has_no_sandbox() -> None:
    result = resolve_effective_execution_environment_for_profile(
        _no_sandbox_profile(),
        _requirement_sandbox(),
        _hosted_provider(),
    )
    assert result.environment is None
    assert result.failure is not None
    assert result.failure.reason is ExecutionEnvironmentResolutionFailureReason.AUTHORITY_VIOLATION


def test_provider_capability_failure_when_exec_not_supported(
    sandbox_session: SandboxSession,
) -> None:
    provider = _local_provider(sandbox_session).model_copy(
        update={
            "supports_sandboxed_exec": False,
            "process_execution": ProcessExecution.DENIED,
        },
    )
    result = resolve_effective_execution_environment_for_profile(
        _sandbox_profile(),
        _requirement_sandbox(),
        provider,
    )
    assert result.failure is not None
    assert result.failure.reason is ExecutionEnvironmentResolutionFailureReason.PROVIDER_CAPABILITY_UNSATISFIED


def test_missing_provider_fail_closed() -> None:
    result = resolve_effective_execution_environment_for_profile(
        _sandbox_profile(),
        _requirement_sandbox(),
        None,
    )
    assert result.failure is not None
    assert result.failure.reason is ExecutionEnvironmentResolutionFailureReason.PROVIDER_UNAVAILABLE


def test_optional_non_sandbox_operation_without_profile_sandbox() -> None:
    result = resolve_effective_execution_environment(
        profile_authority=profile_isolation_authority(_no_sandbox_profile()),
        requirement=ExecutionEnvironmentRequirement.none(),
        provider_capabilities=None,
    )
    assert result.failure is None
    assert result.environment is not None
    assert result.environment.sandbox_required is False


def test_deterministic_resolution_same_inputs(sandbox_session: SandboxSession) -> None:
    provider = _local_provider(sandbox_session)
    profile = _sandbox_profile()
    requirement = _requirement_sandbox()
    first = resolve_effective_execution_environment_for_profile(profile, requirement, provider)
    second = resolve_effective_execution_environment_for_profile(profile, requirement, provider)
    assert first == second


def test_pinned_revision_r1_x1_r2_x2(sandbox_session: SandboxSession) -> None:
    scope = EffectiveProfileRevisionScope(application_id="p1-8", tenant_id="tenant-a")
    store = InMemoryEffectiveProfileRevisionStore()
    r1 = materialize_effective_profile_revision(
        resolve_profile(_sandbox_profile()),
        scope=scope,
        store=store,
    )
    r2 = materialize_effective_profile_revision(
        resolve_profile(_no_sandbox_profile()),
        scope=scope,
        store=store,
    )
    provider = _local_provider(sandbox_session)
    x1 = resolve_effective_execution_environment_for_revision(
        r1,
        _requirement_sandbox(),
        provider,
    )
    x2 = resolve_effective_execution_environment_for_revision(
        r2,
        _requirement_sandbox(),
        provider,
    )
    assert x1.environment is not None
    assert x2.failure is not None
    assert x2.failure.reason is ExecutionEnvironmentResolutionFailureReason.AUTHORITY_VIOLATION


def test_sandbox_exec_adopts_effective_environment_and_executes(
    sandbox_session: SandboxSession,
) -> None:
    ctx = ToolWiringContext(
        sandbox_session=sandbox_session,
        extras={"effective_environment_profile": _sandbox_profile()},
    )
    out = sandbox_exec(
        ctx,
        SandboxExecInput(operation="echo", payload={"message": "p1-8"}),
    )
    assert out.success is True
    assert out.output.get("message") == "p1-8"


def test_sandbox_exec_fail_closed_without_provider() -> None:
    ctx = ToolWiringContext(extras={"effective_environment_profile": _sandbox_profile()})
    _, error = resolve_tool_execution_environment(ctx, contract=sandbox_exec_contract())
    assert error is not None
    assert error.error == "execution_environment_provider_unavailable"


def test_sandbox_exec_fail_closed_provider_without_authority(
    sandbox_session: SandboxSession,
) -> None:
    ctx = ToolWiringContext(sandbox_session=sandbox_session)
    with patch.object(sandbox_session, "execute") as session_exec:
        with patch("intergrax.runtime.sandbox.session.subprocess.run") as host_exec:
            out = sandbox_exec(
                ctx,
                SandboxExecInput(operation="echo", payload={"message": "no-authority"}),
            )
    session_exec.assert_not_called()
    host_exec.assert_not_called()
    assert out.success is False
    assert out.error == "execution_environment_authority_unavailable"


def test_hosted_provider_fail_closed_without_authority() -> None:
    backend = MagicMock()
    backend.create_session.return_value = HostSession(session_id="remote-no-auth")
    backend.exec.return_value = SandboxExecResult(exit_code=0, stdout="remote-ok", stderr="")
    session = HostedSandboxSession.open(backend, tenant_id="tenant-a", task_id="task-a")
    ctx = ToolWiringContext(sandbox_session=session)
    with patch.object(session, "execute") as session_exec:
        out = sandbox_exec(
            ctx,
            SandboxExecInput(operation="echo", payload={"message": "remote"}),
        )
    session_exec.assert_not_called()
    backend.exec.assert_not_called()
    assert out.success is False
    assert out.error == "execution_environment_authority_unavailable"


def test_pinned_revision_precedence_over_legacy_profile(
    sandbox_session: SandboxSession,
) -> None:
    scope = EffectiveProfileRevisionScope(application_id="p1-8", tenant_id="tenant-a")
    store = InMemoryEffectiveProfileRevisionStore()
    r1_deny = materialize_effective_profile_revision(
        resolve_profile(_no_sandbox_profile()),
        scope=scope,
        store=store,
    )
    ctx = ToolWiringContext(
        sandbox_session=sandbox_session,
        extras={
            "effective_profile_revision": r1_deny,
            "effective_environment_profile": _sandbox_profile(),
        },
    )
    with patch.object(sandbox_session, "execute") as session_exec:
        out = sandbox_exec(
            ctx,
            SandboxExecInput(operation="echo", payload={"message": "deny"}),
        )
    session_exec.assert_not_called()
    assert out.success is False
    assert out.error == "execution_environment_authority_violation"


def test_enforcement_never_synthesizes_profile_authority_from_provider() -> None:
    source = inspect.getsource(enforcement_module)
    assert "ProfileIsolationAuthority(" not in source
    assert "substrate_authority" not in source


def test_sandbox_exec_no_host_subprocess_fallback_when_resolution_fails(
    sandbox_session: SandboxSession,
) -> None:
    ctx = ToolWiringContext(
        sandbox_session=sandbox_session,
        extras={"effective_environment_profile": _no_sandbox_profile()},
    )
    with patch("intergrax.runtime.sandbox.session.subprocess.run") as host_exec:
        out = sandbox_exec(
            ctx,
            SandboxExecInput(operation="run_python", payload={"code": "print('host')"}),
        )
    host_exec.assert_not_called()
    assert out.success is False
    assert out.error == "execution_environment_authority_violation"


def test_local_provider_adapter_conformance(sandbox_session: SandboxSession) -> None:
    caps = capabilities_from_local_session(sandbox_session)
    assert caps.provider_ref.provider_kind is ExecutionEnvironmentProviderKind.LOCAL
    assert caps.supports_workspace_write is True
    assert caps.supports_sandboxed_exec is True


def test_remote_provider_adapter_conformance() -> None:
    backend = MagicMock()
    backend.security_capabilities.return_value = MagicMock(
        provider_id="hosted:fake",
        network_egress_deny_enforced=None,
    )
    caps = capabilities_from_host_backend(backend)
    assert caps.provider_ref.provider_kind is ExecutionEnvironmentProviderKind.HOSTED
    assert caps.supports_sandboxed_exec is True
    assert caps.network_access is NetworkAccess.NONE
    assert caps.supports_network_isolation is None


def test_remote_hosted_session_exec_without_host_fallback() -> None:
    backend = MagicMock()
    backend.create_session.return_value = HostSession(session_id="remote-1")
    backend.exec.return_value = SandboxExecResult(exit_code=0, stdout="remote-ok", stderr="")
    session = HostedSandboxSession.open(backend, tenant_id="tenant-a", task_id="task-a")
    ctx = ToolWiringContext(
        sandbox_session=session,
        extras={"effective_environment_profile": _sandbox_profile()},
    )
    out = sandbox_exec(
        ctx,
        SandboxExecInput(operation="echo", payload={"message": "remote"}),
    )
    assert out.success is True
    backend.exec.assert_called_once()
    with patch("intergrax.runtime.sandbox.session.subprocess.run") as host_exec:
        sandbox_exec(
            ctx,
            SandboxExecInput(operation="echo", payload={"message": "remote"}),
        )
    host_exec.assert_not_called()


def test_tenant_scoped_local_provider_probe(sandbox_session: SandboxSession) -> None:
    ctx_a = ToolWiringContext(sandbox_session=sandbox_session)
    other = SandboxSession.create(
        sandbox_session.root.parent,
        tenant_id="tenant-b",
        task_id="task-b",
    )
    ctx_b = ToolWiringContext(sandbox_session=other)
    caps_a = probe_provider_capabilities_from_wiring(ctx_a)
    caps_b = probe_provider_capabilities_from_wiring(ctx_b)
    assert caps_a[0].provider_ref.provider_id != caps_b[0].provider_ref.provider_id


def test_child_narrowing_allowed_and_broader_denied() -> None:
    parent = _requirement_sandbox()
    narrower = ExecutionEnvironmentRequirement(
        sandbox_required=True,
        filesystem_access=FilesystemAccess.READ_ONLY,
        network_access=NetworkAccess.NONE,
        process_execution=ProcessExecution.SANDBOXED,
    )
    broader = ExecutionEnvironmentRequirement(
        sandbox_required=True,
        filesystem_access=FilesystemAccess.WORKSPACE_WRITE,
        network_access=NetworkAccess.ALLOWED,
        process_execution=ProcessExecution.SANDBOXED,
        privilege_mode=PrivilegeMode.PRIVILEGED,
    )
    assert validate_child_requirement_not_widening(parent, narrower) is None
    failure = validate_child_requirement_not_widening(narrower, broader)
    assert failure is not None
    assert failure.reason is ExecutionEnvironmentResolutionFailureReason.AUTHORITY_VIOLATION


def test_provider_disappearance_fails_closed_on_next_operation(
    sandbox_session: SandboxSession,
) -> None:
    ctx = ToolWiringContext(
        sandbox_session=sandbox_session,
        extras={"effective_environment_profile": _sandbox_profile()},
    )
    assert sandbox_exec(
        ctx,
        SandboxExecInput(operation="echo", payload={"message": "ok"}),
    ).success is True
    ctx_after = ToolWiringContext(extras={"effective_environment_profile": _sandbox_profile()})
    out = sandbox_exec(
        ctx_after,
        SandboxExecInput(operation="echo", payload={"message": "fail"}),
    )
    assert out.success is False
    assert "sandbox_session_not_configured" in out.error or out.error.startswith(
        "execution_environment_",
    )


def test_runtime_inspection_exposes_effective_environment() -> None:
    scope = EffectiveProfileRevisionScope(application_id="p1-8", tenant_id="tenant-a")
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    revision = materialize_effective_profile_revision(
        resolve_profile(_sandbox_profile()),
        scope=scope,
        store=revision_store,
    )
    execution_id = mint_execution_id()
    pin_effective_profile_revision_for_execution(
        revision=revision,
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store,
        revision_store=revision_store,
    )
    service = RuntimeInspectionService(
        revision_store=revision_store,
        pinning_store=pinning_store,
    )
    result = service.inspect_execution(
        tenant_id="tenant-a",
        execution_id=execution_id,
        scope_application_id="p1-8",
        scope_tenant_id="tenant-a",
    )
    kinds = {item.subject for item in result.extension_evidence}
    assert "execution_effective_environment" in kinds
    payload = next(
        item.payload
        for item in result.extension_evidence
        if item.subject == "execution_effective_environment"
    )
    assert payload["status"] in {"resolved", "unavailable"}
    dumped = result.model_dump(mode="json")
    assert "sandbox_session" not in str(dumped)
    assert "SandboxHostBackend" not in str(dumped)


def test_sandbox_configured_does_not_imply_tool_authority() -> None:
    profile = _sandbox_profile(enable_exec_tool=False)
    authority = profile_isolation_authority(profile)
    assert authority.sandbox_configured is True
    assert authority.process_execution is ProcessExecution.SANDBOXED
    enabled = set(profile.tool_profile.enabled)
    assert "sandbox.exec" not in enabled


def test_profile_authority_maps_only_sandbox_profile_contract_fields() -> None:
    profile = _sandbox_profile()
    authority = profile_isolation_authority(profile)
    assert profile.sandbox is not None
    assert authority.sandbox_configured is True
    assert authority.filesystem_access is FilesystemAccess.WORKSPACE_WRITE
    assert authority.network_access is NetworkAccess.NONE
    assert authority.process_execution is ProcessExecution.SANDBOXED
    assert authority.privilege_mode is PrivilegeMode.STANDARD
