# © Artur Czarnecki. All rights reserved.

"""TR-01-RQ-C1B — trusted provider capability attestation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from intergrax.contracts.runtime_sandbox_isolation_authority import (
    RuntimeSandboxIsolationAuthority,
)
from intergrax.contracts.sandbox_profile import SandboxProfile as ContractSandboxProfile
from intergrax.integrations.contracts.sandbox_host import SandboxExecResult, SandboxSession as HostSession
from intergrax.runtime.sandbox.contracts import (
    SandboxExecCapable,
    SandboxSecurityCapabilities,
    SandboxSecurityCapable,
)
from intergrax.runtime.sandbox.enforcement import resolve_tool_execution_environment
from intergrax.runtime.sandbox.execution_environment import (
    ExecutionEnvironmentProviderKind,
    ExecutionEnvironmentResolutionFailureReason,
    FilesystemAccess,
    ProcessExecution,
)
from intergrax.runtime.sandbox.hosted_session import HostedSandboxSession
from intergrax.runtime.sandbox.provider_adapters import (
    capabilities_from_attested_exec_session,
    capabilities_from_security_attestation,
    probe_provider_capabilities_from_wiring,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.bundles import IsolationBundle
from intergrax.applications.contracts.environment_profile.sub_profiles import SandboxProfile
from intergrax.runtime.sandbox.execution_environment import ExecutionEnvironmentRequirement
from intergrax.runtime.sandbox.resolver import resolve_effective_execution_environment_for_profile
from intergrax.tools.core.contracts import ToolIsolationRequirement
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.tools.providers.sandbox.bundle import sandbox_exec_contract
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]


class _PlainExecOnly:
    session_id = "plain-exec"

    def execute(self, operation: str, payload: dict | None = None) -> object:
        return MagicMock(output={}, session_id=self.session_id)


class _AttestedExternalProvider:
    session_id = "external-attested"

    def execute(self, operation: str, payload: dict | None = None) -> object:
        return MagicMock(output={"operation": operation}, session_id=self.session_id)

    def security_capabilities(self) -> SandboxSecurityCapabilities:
        return SandboxSecurityCapabilities(
            isolation_tier="local",
            provider_id="external:plugin-1",
            network_egress_deny_enforced=True,
            network_egress_allowlist_enforced=None,
        )


def _explicit_authority() -> RuntimeSandboxIsolationAuthority:
    return RuntimeSandboxIsolationAuthority(
        sandbox=ContractSandboxProfile(enable_exec_tool=True),
    )


def test_cap_c1b_1_plain_exec_does_not_prove_sandboxed_process() -> None:
    caps = capabilities_from_attested_exec_session(_PlainExecOnly())
    assert caps is None


def test_cap_c1b_2_plain_exec_does_not_prove_workspace_write() -> None:
    ctx = ToolWiringContext(sandbox_session=_PlainExecOnly())
    assert probe_provider_capabilities_from_wiring(ctx) == ()


def test_cap_c1b_3_plain_exec_no_fabricated_local_provider_kind() -> None:
    providers = probe_provider_capabilities_from_wiring(
        ToolWiringContext(sandbox_session=_PlainExecOnly()),
    )
    assert all(
        p.provider_ref.provider_kind is not ExecutionEnvironmentProviderKind.LOCAL
        for p in providers
    )


def test_cap_c1b_4_plain_provider_fails_isolation_resolution() -> None:
    ctx = ToolWiringContext(
        sandbox_session=_PlainExecOnly(),
        sandbox_isolation_authority=_explicit_authority(),
    )
    _, err = resolve_tool_execution_environment(ctx, contract=sandbox_exec_contract())
    assert err is not None
    assert err.error == "execution_environment_provider_unavailable"


def test_cap_c1b_5_attested_external_provider_succeeds_when_evidence_sufficient() -> None:
    provider = _AttestedExternalProvider()
    assert isinstance(provider, SandboxExecCapable)
    assert isinstance(provider, SandboxSecurityCapable)
    ctx = ToolWiringContext(
        sandbox_session=provider,
        sandbox_isolation_authority=_explicit_authority(),
    )
    env, err = resolve_tool_execution_environment(ctx, contract=sandbox_exec_contract())
    assert err is None
    assert env is not None


def test_cap_c1b_6_insufficient_security_evidence_fails_closed() -> None:
    security = SandboxSecurityCapabilities(
        isolation_tier="local",
        provider_id="external:weak",
    )
    caps = capabilities_from_security_attestation(
        security,
        supports_sandboxed_exec=False,
        supports_workspace_write=False,
        filesystem_access=FilesystemAccess.NONE,
        process_execution=ProcessExecution.DENIED,
    )
    result = resolve_effective_execution_environment_for_profile(
        _explicit_authority(),
        ExecutionEnvironmentRequirement.from_tool_isolation(ToolIsolationRequirement.SANDBOX),
        caps,
    )
    assert result.failure is not None
    assert (
        result.failure.reason
        is ExecutionEnvironmentResolutionFailureReason.PROVIDER_CAPABILITY_UNSATISFIED
    )


def test_cap_c1b_7_local_sandbox_session_remains_green(sandbox_session: SandboxSession) -> None:
    ctx = ToolWiringContext(
        sandbox_session=sandbox_session,
        sandbox_isolation_authority=_explicit_authority(),
    )
    env, err = resolve_tool_execution_environment(ctx, contract=sandbox_exec_contract())
    assert err is None
    assert env is not None


def test_cap_c1b_8_hosted_session_remains_green() -> None:
    backend = MagicMock()
    backend.create_session.return_value = HostSession(session_id="remote-cap")
    backend.exec.return_value = SandboxExecResult(exit_code=0, stdout="ok", stderr="")
    session = HostedSandboxSession.open(backend, tenant_id="t", task_id="task")
    profile = ApplicationEnvironmentProfile.lab_defaults(profile_id="hosted-cap").model_copy(
        update={
            "isolation": IsolationBundle(
                sandbox=SandboxProfile(enable_exec_tool=True),
            ),
        },
    )
    ctx = ToolWiringContext(
        sandbox_session=session,
        extras={"effective_environment_profile": profile},
    )
    env, err = resolve_tool_execution_environment(ctx, contract=sandbox_exec_contract())
    assert err is None
    assert env is not None


def test_cap_c1b_static_no_capabilities_from_exec_capable_session() -> None:
    source = (_REPO_ROOT / "intergrax/runtime/sandbox/provider_adapters.py").read_text(
        encoding="utf-8",
    )
    assert "capabilities_from_exec_capable_session" not in source


@pytest.fixture
def sandbox_session(tmp_path: Path) -> SandboxSession:
    return SandboxSession.create(tmp_path, tenant_id="tenant-1", task_id="task-1")
