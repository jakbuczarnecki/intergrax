# © Artur Czarnecki. All rights reserved.

"""P0-SAFETY-7 — sandbox / isolation fail-closed conformance."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.sandbox.isolation_errors import (
    SandboxIsolationFailureReason,
    SandboxIsolationRequiredError,
)
from intergrax.runtime.sandbox.isolation_gate import (
    SandboxIsolationAvailability,
    require_sandbox_isolation,
    sandbox_availability_from_wiring,
    sandbox_availability_provider,
)
from intergrax.runtime.sandbox.sandbox_runtime import requires_sandbox_tool
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.providers.sandbox.bundle import register_sandbox_tools
from intergrax.tools.providers.sandbox.contracts import CodeExecInput, SandboxExecInput
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from testing_support.builder import build_runtime_state_for_tests, canonical_execution_identity_scope

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_RUN_ID = "p0-safety-7-run"
_AGENT_ID = "p0-safety-7-agent"


class _EchoInput(BaseModel):
    message: str


class _EchoOutput(BaseModel):
    message: str


class CountingExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
        self.calls += 1
        return _EchoOutput(message="ok")


def _non_sandbox_contract() -> ToolContract:
    return ToolContract(
        tool_id="echo.basic",
        name="echo.basic",
        description="echo",
        input_schema=_EchoInput,
        output_schema=_EchoOutput,
        error_mapping={},
        side_effects=False,
        risk_level=ToolRiskLevel.LOW,
    )


@pytest.fixture
def sandbox_session(tmp_path: Path) -> SandboxSession:
    return SandboxSession.create(
        tmp_path,
        tenant_id="tenant-a",
        task_id="task-a",
        allowed_operations=frozenset(
            {"echo", "write_file", "read_file", "list_files", "run_python", "browser_fetch"},
        ),
    )


def test_requires_sandbox_tool_is_canonical_set_not_prefix() -> None:
    assert requires_sandbox_tool("code.exec") is True
    assert requires_sandbox_tool("workspace.write") is False
    assert requires_sandbox_tool("sandbox.workspace.exec") is False


def test_missing_sandbox_fails_before_executor() -> None:
    registry = ToolRegistry()
    ctx = ToolWiringContext()
    register_sandbox_tools(registry, ctx)
    executor = CountingExecutor()
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=executor,
        sandbox_availability=sandbox_availability_provider(ctx),
    )
    state = build_runtime_state_for_tests(run_id=_RUN_ID)
    request = ToolExecutionRequest(
        run_id=_RUN_ID,
        step_id="step/1",
        tool_id="code.exec",
        input=CodeExecInput(code="print(1)", language="python"),
    )

    with pytest.raises(SandboxIsolationRequiredError) as exc_info:
        invoker.invoke(state=state, agent_id=_AGENT_ID, request=request)

    assert exc_info.value.reason is SandboxIsolationFailureReason.NOT_CONFIGURED
    assert executor.calls == 0


def test_missing_sandbox_never_calls_host_subprocess(sandbox_session: SandboxSession) -> None:
    registry = ToolRegistry()
    ctx = ToolWiringContext()
    register_sandbox_tools(registry, ctx)
    executor = CountingExecutor()
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=executor,
        sandbox_availability=sandbox_availability_provider(ctx),
    )
    state = build_runtime_state_for_tests(run_id=_RUN_ID)
    request = ToolExecutionRequest(
        run_id=_RUN_ID,
        step_id="step/1",
        tool_id="code.exec",
        input=CodeExecInput(code="print(1)", language="python"),
    )

    with patch("subprocess.run") as subprocess_run:
        with pytest.raises(SandboxIsolationRequiredError):
            invoker.invoke(state=state, agent_id=_AGENT_ID, request=request)
        subprocess_run.assert_not_called()
    assert executor.calls == 0


def test_unhealthy_sandbox_fails_closed() -> None:
    ctx = ToolWiringContext(sandbox_session=MagicMock())
    availability = sandbox_availability_from_wiring(ctx, healthy=False)
    with pytest.raises(SandboxIsolationRequiredError) as exc_info:
        require_sandbox_isolation(
            tool_id="sandbox.exec",
            availability=availability,
            run_id=_RUN_ID,
            agent_id=_AGENT_ID,
        )
    assert exc_info.value.reason is SandboxIsolationFailureReason.UNHEALTHY


def test_valid_sandbox_reaches_provider(sandbox_session: SandboxSession) -> None:
    registry = ToolRegistry()
    ctx = ToolWiringContext(sandbox_session=sandbox_session)
    register_sandbox_tools(registry, ctx)
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
        sandbox_availability=sandbox_availability_provider(ctx),
    )
    state = build_runtime_state_for_tests(run_id=_RUN_ID)
    request = ToolExecutionRequest(
        run_id=state.run_id,
        step_id="step/1",
        tool_id="sandbox.exec",
        input=SandboxExecInput(operation="echo", payload={"message": "ok"}),
    )

    with canonical_execution_identity_scope(_RUN_ID):
        with patch(
            "intergrax.runtime.nexus.tools.invoker.require_meaningful_side_effect_authorization",
        ):
            result = invoker.invoke(state=state, agent_id=_AGENT_ID, request=request)

    assert result.success is True
    assert result.output is not None
    assert result.output.success is True


def test_non_isolated_tool_works_without_sandbox() -> None:
    registry = ToolRegistry()
    registry.register(_non_sandbox_contract(), CountingExecutor())
    ctx = ToolWiringContext()
    executor = CountingExecutor()
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=executor,
        sandbox_availability=sandbox_availability_provider(ctx),
    )
    state = build_runtime_state_for_tests(run_id=_RUN_ID)
    request = ToolExecutionRequest(
        run_id=state.run_id,
        step_id="step/1",
        tool_id="echo.basic",
        input=_EchoInput(message="hi"),
    )

    with canonical_execution_identity_scope(_RUN_ID):
        result = invoker.invoke(state=state, agent_id=_AGENT_ID, request=request)

    assert result.success is True
    assert executor.calls == 1


def test_plugin_like_isolated_tool_uses_same_gate() -> None:
    registry = ToolRegistry()
    plugin_contract = ToolContract(
        tool_id="codecraft.run",
        name="codecraft.run",
        description="plugin-like isolated tool",
        input_schema=CodeExecInput,
        output_schema=_EchoOutput,
        error_mapping={},
        side_effects=True,
        risk_level=ToolRiskLevel.HIGH,
        category="sandbox",
    )
    executor = CountingExecutor()
    registry.register(plugin_contract, executor)
    ctx = ToolWiringContext()
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=executor,
        sandbox_availability=sandbox_availability_provider(ctx),
    )
    state = build_runtime_state_for_tests(run_id=_RUN_ID)
    request = ToolExecutionRequest(
        run_id=_RUN_ID,
        step_id="step/1",
        tool_id="codecraft.run",
        input=CodeExecInput(code="print(1)", language="python"),
    )

    with pytest.raises(SandboxIsolationRequiredError):
        invoker.invoke(state=state, agent_id=_AGENT_ID, request=request)
    assert executor.calls == 0


def test_container_tier_without_hosted_backend_fails_closed(sandbox_session: SandboxSession) -> None:
    from intergrax.codecraft.profile import CodeCraftProfile
    from intergrax.runtime.codecraft.sandbox_resolver import resolve_craft_sandbox_session

    profile = CodeCraftProfile(mode="autonomous", isolation_tier="container")
    ctx = ToolWiringContext(sandbox_session=sandbox_session)
    resolved = resolve_craft_sandbox_session(ctx, profile, tenant_id="t", task_id="task")
    assert resolved is None


def test_cloud_tier_without_hosted_backend_fails_closed(sandbox_session: SandboxSession) -> None:
    from intergrax.codecraft.profile import CodeCraftProfile
    from intergrax.runtime.codecraft.sandbox_resolver import resolve_craft_sandbox_session

    profile = CodeCraftProfile(mode="autonomous", isolation_tier="cloud")
    ctx = ToolWiringContext(sandbox_session=sandbox_session)
    resolved = resolve_craft_sandbox_session(ctx, profile, tenant_id="t", task_id="task")
    assert resolved is None


def test_isolation_requirement_not_weakened_by_missing_task_flag(sandbox_session: SandboxSession) -> None:
    availability = SandboxIsolationAvailability(
        session_configured=False,
        host_configured=False,
    )
    with pytest.raises(SandboxIsolationRequiredError):
        require_sandbox_isolation(
            tool_id="script.run",
            availability=availability,
            run_id=_RUN_ID,
            agent_id=_AGENT_ID,
        )
