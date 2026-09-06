# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.bundles import IsolationBundle
from intergrax.applications.contracts.environment_profile.sub_profiles import SandboxProfile
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.providers.sandbox.bundle import register_sandbox_tools
from intergrax.tools.providers.sandbox.contracts import SandboxExecInput
from intergrax.tools.providers.sandbox.service import sandbox_exec
from intergrax.tools.registry.bootstrap import register_default_tools, reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog, get_bundle, list_catalog_tool_ids
from intergrax.tools.registry.factory import build_registry_from_profile
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.runtime.sandbox.isolation_gate import sandbox_availability_provider
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from testing_support.builder import build_runtime_state_for_tests

pytestmark = pytest.mark.unit


def _sandbox_env_profile() -> ApplicationEnvironmentProfile:
    profile = ApplicationEnvironmentProfile.lab_defaults(profile_id="sandbox-exec-test")
    return profile.model_copy(
        update={
            "isolation": IsolationBundle(
                sandbox=SandboxProfile(enable_exec_tool=True),
            ),
        },
    )


def _sandbox_ctx(session: SandboxSession) -> ToolWiringContext:
    return ToolWiringContext(
        sandbox_session=session,
        extras={"effective_environment_profile": _sandbox_env_profile()},
    )


@pytest.fixture
def sandbox_session(tmp_path: Path) -> SandboxSession:
    return SandboxSession.create(
        tmp_path,
        tenant_id="tenant-1",
        task_id="task-1",
    )


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_tool_catalog()
    reset_default_tools_bootstrap()
    yield
    clear_tool_catalog()
    reset_default_tools_bootstrap()


def test_sandbox_exec_echo(sandbox_session: SandboxSession) -> None:
    ctx = _sandbox_ctx(sandbox_session)
    out = sandbox_exec(
        ctx,
        SandboxExecInput(operation="echo", payload={"message": "hello sandbox"}),
    )
    assert out.success is True
    assert out.output.get("message") == "hello sandbox"
    assert out.session_id == sandbox_session.session_id


def test_sandbox_exec_not_configured() -> None:
    out = sandbox_exec(
        ToolWiringContext(),
        SandboxExecInput(operation="echo", payload={"text": "x"}),
    )
    assert out.success is False
    assert out.error == "execution_environment_authority_unavailable"


def test_sandbox_tool_registered_in_catalog() -> None:
    register_default_tools()
    assert "sandbox.exec" in list_catalog_tool_ids()
    assert get_bundle("sandbox").tool_ids == (
        "sandbox.exec",
        "code.exec",
        "script.run",
        "browser.run",
        "sandbox.list_operations",
    )


def test_sandbox_exec_via_runtime_invoker(sandbox_session: SandboxSession) -> None:
    ctx = _sandbox_ctx(sandbox_session)
    registry = ToolRegistry()
    register_sandbox_tools(registry, ctx)

    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
        sandbox_availability=sandbox_availability_provider(ctx),
    )
    state = build_runtime_state_for_tests(run_id="sbox_run")
    request = ToolExecutionRequest(
        run_id="sbox_run",
        step_id="step/1",
        tool_id="sandbox.exec",
        input=SandboxExecInput(operation="echo", payload={"message": "via invoker"}),
    )

    result = invoker.invoke(state=state, agent_id="agent", request=request)

    assert result.success is True
    assert result.output is not None
    assert result.output.success is True


def test_build_registry_enables_sandbox_tool(sandbox_session: SandboxSession) -> None:
    register_default_tools()
    ctx = _sandbox_ctx(sandbox_session)
    registry = build_registry_from_profile(ToolProfile(enabled=["sandbox.exec"]), ctx=ctx)
    assert registry.has("sandbox.exec")
