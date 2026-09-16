# © Artur Czarnecki. All rights reserved.

"""TR-01-RQ-C1A — explicit sandbox isolation authority vs session availability."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.profile_resolution import (
    InMemoryEffectiveProfileRevisionStore,
    materialize_effective_profile_revision,
    resolve_profile,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.bundles import IsolationBundle
from intergrax.applications.contracts.environment_profile.sub_profiles import SandboxProfile
from intergrax.applications.contracts.profile_resolution import EffectiveProfileRevisionScope
from intergrax.contracts.runtime_sandbox_isolation_authority import (
    RuntimeSandboxIsolationAuthority,
)
from intergrax.contracts.sandbox_profile import SandboxProfile as ContractSandboxProfile
from intergrax.runtime.nexus.tools.uaep_tool_gateway import BoundToolGateway
from intergrax.runtime.sandbox.enforcement import resolve_tool_execution_environment
from intergrax.runtime.sandbox.execution_environment import ExecutionEnvironmentResolutionFailureReason
from intergrax.runtime.sandbox.resolver import (
    profile_isolation_authority,
    validate_child_requirement_not_widening,
)
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.tools.invocation_wiring import ToolInvocationWiring
from intergrax.tools.invocation_wiring_adapter import merge_invocation_wiring
from intergrax.tools.providers.sandbox.bundle import register_sandbox_tools, sandbox_exec_contract
from intergrax.tools.providers.sandbox.contracts import SandboxExecInput
from intergrax.tools.providers.sandbox.service import sandbox_exec
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]


class _CustomSandboxExecCapable:
    session_id = "custom-capable"

    def execute(self, operation: str, payload: dict) -> object:
        return MagicMock(output={"operation": operation, "payload": payload}, session_id=self.session_id)


def _app_profile(*, enable: bool = True) -> ApplicationEnvironmentProfile:
    profile = ApplicationEnvironmentProfile.lab_defaults(profile_id="auth-c1a")
    if not enable:
        return profile.model_copy(update={"isolation": IsolationBundle()})
    return profile.model_copy(
        update={
            "isolation": IsolationBundle(
                sandbox=SandboxProfile(enable_exec_tool=enable),
            ),
        },
    )


def _explicit_authority(*, enable: bool = True) -> RuntimeSandboxIsolationAuthority:
    return RuntimeSandboxIsolationAuthority(
        sandbox=ContractSandboxProfile(enable_exec_tool=enable),
    )


@pytest.fixture
def sandbox_session(tmp_path: Path) -> SandboxSession:
    return SandboxSession.create(
        tmp_path,
        tenant_id="tenant-1",
        task_id="task-1",
    )


def test_auth_c1a_1_session_only_does_not_grant_sandbox_authority(
    sandbox_session: SandboxSession,
) -> None:
    merged = merge_invocation_wiring(
        ToolWiringContext(),
        ToolInvocationWiring(sandbox_session=sandbox_session),
    )
    assert merged.sandbox_session is sandbox_session
    assert "effective_environment_profile" not in merged.extras
    assert merged.sandbox_isolation_authority is None
    _, err = resolve_tool_execution_environment(merged, contract=sandbox_exec_contract())
    assert err is not None
    assert err.error == "execution_environment_authority_unavailable"


def test_auth_c1a_static_no_runtime_host_synthetic_profile() -> None:
    assert not (_REPO_ROOT / "intergrax/runtime/sandbox/runtime_host_wiring.py").exists()
    adapter = (_REPO_ROOT / "intergrax/tools/invocation_wiring_adapter.py").read_text(
        encoding="utf-8",
    )
    assert "runtime_host_sandbox" not in adapter
    assert "for_attached_session" not in adapter
    tree = ast.parse(adapter)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "SandboxProfile":
                raise AssertionError("invocation adapter must not construct SandboxProfile")


def test_auth_c1a_2_tool_registration_alone_no_authority(
    sandbox_session: SandboxSession,
) -> None:
    registry = ToolRegistry()
    register_sandbox_tools(registry, ToolWiringContext(sandbox_session=sandbox_session))
    assert registry.has("sandbox.exec")
    out = sandbox_exec(
        ToolWiringContext(sandbox_session=sandbox_session),
        SandboxExecInput(operation="echo", payload={"message": "x"}),
    )
    assert out.success is False
    assert out.error == "execution_environment_authority_unavailable"


def test_auth_c1a_3_toolprofile_alone_no_authority(
    sandbox_session: SandboxSession,
) -> None:
    ctx = ToolWiringContext(sandbox_session=sandbox_session)
    out = sandbox_exec(
        ctx,
        SandboxExecInput(operation="echo", payload={"message": "x"}),
    )
    assert out.success is False
    assert out.error == "execution_environment_authority_unavailable"


def test_auth_c1a_4_allowed_tools_scope_does_not_grant_authority(
    sandbox_session: SandboxSession,
) -> None:
    del sandbox_session  # scope declaration only — no session authority mint
    ctx = ToolWiringContext(
        extras={"agent_allowed_tools": ("sandbox.exec",)},
    )
    _, err = resolve_tool_execution_environment(ctx, contract=sandbox_exec_contract())
    assert err is not None


def test_auth_c1a_5_explicit_authority_without_session_fails_closed() -> None:
    ctx = ToolWiringContext(
        sandbox_isolation_authority=_explicit_authority(),
    )
    _, err = resolve_tool_execution_environment(ctx, contract=sandbox_exec_contract())
    assert err is not None
    assert err.error == "execution_environment_provider_unavailable"


def test_auth_c1a_6_explicit_authority_plus_session_allows_resolution(
    sandbox_session: SandboxSession,
) -> None:
    ctx = ToolWiringContext(
        sandbox_session=sandbox_session,
        sandbox_isolation_authority=_explicit_authority(),
    )
    env, err = resolve_tool_execution_environment(ctx, contract=sandbox_exec_contract())
    assert err is None
    assert env is not None


def test_auth_c1a_8_parent_authority_deny_cannot_widen_downstream() -> None:
    from intergrax.runtime.sandbox.execution_environment import ExecutionEnvironmentRequirement
    from intergrax.tools.core.contracts import ToolIsolationRequirement

    parent = ExecutionEnvironmentRequirement.none()
    child = ExecutionEnvironmentRequirement.from_tool_isolation(ToolIsolationRequirement.SANDBOX)
    failure = validate_child_requirement_not_widening(parent, child)
    assert failure is not None
    assert failure.reason is ExecutionEnvironmentResolutionFailureReason.AUTHORITY_VIOLATION


def test_auth_c1a_9_pinned_effective_profile_revision_wins_over_compat() -> None:
    pinned_profile = _app_profile(enable=True)
    disabled_compat = _app_profile(enable=False)
    store = InMemoryEffectiveProfileRevisionStore()
    revision = materialize_effective_profile_revision(
        resolve_profile(pinned_profile),
        scope=EffectiveProfileRevisionScope(application_id="auth-c1a", tenant_id="t1"),
        store=store,
    )
    ctx = ToolWiringContext(
        extras={
            "effective_profile_revision": revision,
            "effective_environment_profile": disabled_compat,
        },
    )
    from intergrax.runtime.sandbox.enforcement import _profile_from_context

    selected = _profile_from_context(ctx)
    assert selected is revision.effective_profile
    assert profile_isolation_authority(selected).sandbox_configured is True
    assert profile_isolation_authority(disabled_compat).sandbox_configured is False


def test_auth_c1a_10_custom_sandbox_exec_capable_with_explicit_authority() -> None:
    ctx = ToolWiringContext(
        sandbox_session=_CustomSandboxExecCapable(),
        sandbox_isolation_authority=_explicit_authority(),
    )
    _, err = resolve_tool_execution_environment(ctx, contract=sandbox_exec_contract())
    assert err is not None
    assert err.error == "execution_environment_provider_unavailable"


def test_auth_c1a_12_bound_tool_gateway_facade_only() -> None:
    source = (_REPO_ROOT / "intergrax/runtime/nexus/tools/uaep_tool_gateway.py").read_text(
        encoding="utf-8",
    )
    assert "session.execute(" not in source
    assert isinstance(BoundToolGateway, type)


def test_auth_c1a_13_no_synthetic_sandboxprofile_from_session_in_adapter() -> None:
    test_auth_c1a_static_no_runtime_host_synthetic_profile()


def test_session_only_merge_does_not_set_effective_environment_profile(
    sandbox_session: SandboxSession,
) -> None:
    merged = merge_invocation_wiring(
        ToolWiringContext(extras={"marker": "keep"}),
        ToolInvocationWiring(sandbox_session=sandbox_session),
    )
    assert merged.extras.get("marker") == "keep"
    assert "effective_environment_profile" not in merged.extras


def test_explicit_authority_plus_session_allows_resolution(
    sandbox_session: SandboxSession,
) -> None:
    test_auth_c1a_6_explicit_authority_plus_session_allows_resolution(sandbox_session)
