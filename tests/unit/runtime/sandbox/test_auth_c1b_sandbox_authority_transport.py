# © Artur Czarnecki. All rights reserved.

"""TR-01-RQ-C1B — typed sandbox isolation authority transport."""

from __future__ import annotations

import ast
from pathlib import Path

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
from intergrax.contracts.runtime_sandbox_isolation_authority import RuntimeSandboxIsolationAuthority
from intergrax.tools.registry.sandbox_isolation_wiring import apply_runtime_sandbox_isolation_authority
from intergrax.contracts.sandbox_profile import SandboxProfile as ContractSandboxProfile
from intergrax.runtime.sandbox.enforcement import _profile_from_context, resolve_tool_execution_environment
from intergrax.runtime.sandbox.resolver import profile_isolation_authority
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.tools.providers.sandbox.bundle import sandbox_exec_contract
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _explicit_authority(*, enable: bool = True) -> RuntimeSandboxIsolationAuthority:
    return RuntimeSandboxIsolationAuthority(
        sandbox=ContractSandboxProfile(enable_exec_tool=enable),
    )


def _app_profile(*, enable: bool = True) -> ApplicationEnvironmentProfile:
    profile = ApplicationEnvironmentProfile.lab_defaults(profile_id="auth-c1b")
    if not enable:
        return profile.model_copy(update={"isolation": IsolationBundle()})
    return profile.model_copy(
        update={
            "isolation": IsolationBundle(
                sandbox=SandboxProfile(enable_exec_tool=enable),
            ),
        },
    )


def test_auth_c1b_1_runtime_authority_typed_field_on_wiring_context() -> None:
    ctx = ToolWiringContext(sandbox_isolation_authority=_explicit_authority())
    assert ctx.sandbox_isolation_authority is not None
    assert _profile_from_context(ctx) is ctx.sandbox_isolation_authority


def test_auth_c1b_2_no_runtime_sandbox_isolation_authority_extra_key_in_repo() -> None:
    assert "RUNTIME_SANDBOX_ISOLATION_AUTHORITY_EXTRA_KEY" not in (
        _REPO_ROOT / "intergrax/contracts/runtime_sandbox_isolation_authority.py"
    ).read_text(encoding="utf-8")


def test_auth_c1b_3_enforcement_does_not_read_authority_from_extras() -> None:
    source = (_REPO_ROOT / "intergrax/runtime/sandbox/enforcement.py").read_text(encoding="utf-8")
    assert "runtime_sandbox_isolation_authority" not in source
    assert "ctx.sandbox_isolation_authority" in source


def test_auth_c1b_4_no_hasattr_getattr_in_runtime_authority_contract() -> None:
    source = (
        _REPO_ROOT / "intergrax/contracts/runtime_sandbox_isolation_authority.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"hasattr", "getattr"}:
                raise AssertionError("runtime_sandbox_isolation_authority must not use reflection")


def test_auth_c1b_5_pinned_profile_precedence_over_runtime_authority(
    sandbox_session: SandboxSession,
) -> None:
    pinned_profile = _app_profile(enable=False)
    store = InMemoryEffectiveProfileRevisionStore()
    revision = materialize_effective_profile_revision(
        resolve_profile(pinned_profile),
        scope=EffectiveProfileRevisionScope(application_id="auth-c1b", tenant_id="t1"),
        store=store,
    )
    ctx = ToolWiringContext(
        sandbox_session=sandbox_session,
        sandbox_isolation_authority=_explicit_authority(enable=True),
        extras={"effective_profile_revision": revision},
    )
    selected = _profile_from_context(ctx)
    assert selected is revision.effective_profile
    assert profile_isolation_authority(selected).sandbox_configured is False


def test_auth_c1b_6_explicit_runtime_authority_via_composition_helper() -> None:
    base = ToolWiringContext()
    merged = apply_runtime_sandbox_isolation_authority(base, _explicit_authority())
    assert merged.sandbox_isolation_authority is not None
    _, err = resolve_tool_execution_environment(merged, contract=sandbox_exec_contract())
    assert err is not None
    assert err.error == "execution_environment_provider_unavailable"


@pytest.fixture
def sandbox_session(tmp_path: Path) -> SandboxSession:
    return SandboxSession.create(
        tmp_path,
        tenant_id="tenant-1",
        task_id="task-1",
    )


def test_auth_c1b_runtime_authority_plus_local_session_resolves(
    sandbox_session: SandboxSession,
) -> None:
    ctx = ToolWiringContext(
        sandbox_session=sandbox_session,
        sandbox_isolation_authority=_explicit_authority(),
    )
    env, err = resolve_tool_execution_environment(ctx, contract=sandbox_exec_contract())
    assert err is None
    assert env is not None
