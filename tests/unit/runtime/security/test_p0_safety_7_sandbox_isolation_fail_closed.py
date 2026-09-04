# © Artur Czarnecki. All rights reserved.

"""P0-SAFETY-7 / 7A — sandbox / isolation fail-closed conformance."""

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
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.tools.core.contracts import (
    ToolContract,
    ToolIsolationRequirement,
    ToolRiskLevel,
    contract_requires_sandbox_isolation,
)
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.providers.codecraft.bundle import register_codecraft_tools
from intergrax.tools.providers.sandbox.bundle import (
    register_sandbox_tools,
    sandbox_exec_contract,
)
from intergrax.tools.providers.sandbox.contracts import CodeExecInput, SandboxExecInput
from intergrax.tools.providers.sandbox.extended_service import (
    BROWSER_RUN_TOOL_ID,
    CODE_EXEC_TOOL_ID,
)
from intergrax.tools.providers.sandbox.service import SANDBOX_EXEC_TOOL_ID
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


def _non_sandbox_contract(*, tool_id: str = "echo.basic") -> ToolContract:
    return ToolContract(
        tool_id=tool_id,
        name=tool_id,
        description="echo",
        input_schema=_EchoInput,
        output_schema=_EchoOutput,
        error_mapping={},
        side_effects=False,
        risk_level=ToolRiskLevel.LOW,
        isolation_requirement=ToolIsolationRequirement.NONE,
    )


def _isolated_plugin_contract(*, tool_id: str) -> ToolContract:
    return ToolContract(
        tool_id=tool_id,
        name=tool_id,
        description="plugin-like isolated tool",
        input_schema=CodeExecInput,
        output_schema=_EchoOutput,
        error_mapping={},
        side_effects=True,
        risk_level=ToolRiskLevel.HIGH,
        category="plugin",
        isolation_requirement=ToolIsolationRequirement.SANDBOX,
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


def test_core_isolated_tools_declare_contract_requirement() -> None:
    registry = ToolRegistry()
    register_sandbox_tools(registry, ToolWiringContext())
    register_codecraft_tools(registry, ToolWiringContext())

    for tool_id in (SANDBOX_EXEC_TOOL_ID, CODE_EXEC_TOOL_ID, BROWSER_RUN_TOOL_ID):
        contract = registry.get(tool_id).contract
        assert contract.isolation_requirement is ToolIsolationRequirement.SANDBOX
        assert contract_requires_sandbox_isolation(contract) is True

    assert sandbox_exec_contract().isolation_requirement is ToolIsolationRequirement.SANDBOX


def test_name_is_not_isolation_authority() -> None:
    registry = ToolRegistry()
    registry.register(
        _isolated_plugin_contract(tool_id="plugin.safe_api"),
        CountingExecutor(),
    )
    registry.register(
        _non_sandbox_contract(tool_id="sandbox.named_but_nonisolated"),
        CountingExecutor(),
    )

    assert (
        registry.get("plugin.safe_api").contract.isolation_requirement
        is ToolIsolationRequirement.SANDBOX
    )
    assert (
        registry.get("sandbox.named_but_nonisolated").contract.isolation_requirement
        is ToolIsolationRequirement.NONE
    )


def test_registry_preserves_isolation_requirement() -> None:
    registry = ToolRegistry()
    contract = _isolated_plugin_contract(tool_id="plugin.python_exec")
    registry.register(contract, CountingExecutor())

    stored = registry.get("plugin.python_exec").contract
    assert stored.isolation_requirement is ToolIsolationRequirement.SANDBOX


def test_missing_provider_fails_before_executor_for_isolated_tool() -> None:
    registry = ToolRegistry()
    register_sandbox_tools(registry, ToolWiringContext())
    executor = CountingExecutor()
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=executor,
        sandbox_availability=None,
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
    registry = ToolRegistry()
    register_sandbox_tools(registry, ToolWiringContext())
    contract = registry.get(SANDBOX_EXEC_TOOL_ID).contract
    ctx = ToolWiringContext(sandbox_session=MagicMock())
    availability = sandbox_availability_from_wiring(ctx, healthy=False)
    with pytest.raises(SandboxIsolationRequiredError) as exc_info:
        require_sandbox_isolation(
            contract=contract,
            availability=availability,
            run_id=_RUN_ID,
            agent_id=_AGENT_ID,
        )
    assert exc_info.value.reason is SandboxIsolationFailureReason.UNHEALTHY


def test_unhealthy_provider_fails_before_executor() -> None:
    registry = ToolRegistry()
    register_sandbox_tools(registry, ToolWiringContext())
    executor = CountingExecutor()

    def _unhealthy() -> SandboxIsolationAvailability:
        return SandboxIsolationAvailability(
            session_configured=True,
            host_configured=False,
            healthy=False,
        )

    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=executor,
        sandbox_availability=_unhealthy,
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

    assert exc_info.value.reason is SandboxIsolationFailureReason.UNHEALTHY
    assert executor.calls == 0


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
    executor = CountingExecutor()
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=executor,
        sandbox_availability=None,
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


def test_plugin_isolated_tool_denied_without_sandbox() -> None:
    registry = ToolRegistry()
    executor = CountingExecutor()
    registry.register(
        _isolated_plugin_contract(tool_id="plugin.python_exec"),
        executor,
    )
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=executor,
        sandbox_availability=None,
    )
    state = build_runtime_state_for_tests(run_id=_RUN_ID)
    request = ToolExecutionRequest(
        run_id=_RUN_ID,
        step_id="step/1",
        tool_id="plugin.python_exec",
        input=CodeExecInput(code="print(1)", language="python"),
    )

    with pytest.raises(SandboxIsolationRequiredError):
        invoker.invoke(state=state, agent_id=_AGENT_ID, request=request)
    assert executor.calls == 0


def test_plugin_isolated_tool_continues_with_valid_sandbox(sandbox_session: SandboxSession) -> None:
    registry = ToolRegistry()
    executor = CountingExecutor()
    registry.register(
        _isolated_plugin_contract(tool_id="plugin.python_exec"),
        executor,
    )
    ctx = ToolWiringContext(sandbox_session=sandbox_session)

    def _healthy() -> SandboxIsolationAvailability:
        return SandboxIsolationAvailability(
            session_configured=True,
            host_configured=False,
            healthy=True,
        )

    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=executor,
        sandbox_availability=_healthy,
    )
    state = build_runtime_state_for_tests(run_id=_RUN_ID)
    request = ToolExecutionRequest(
        run_id=_RUN_ID,
        step_id="step/1",
        tool_id="plugin.python_exec",
        input=CodeExecInput(code="print(1)", language="python"),
    )

    with canonical_execution_identity_scope(_RUN_ID):
        with patch(
            "intergrax.runtime.nexus.tools.invoker.require_meaningful_side_effect_authorization",
        ):
            result = invoker.invoke(state=state, agent_id=_AGENT_ID, request=request)

    assert result.success is True
    assert executor.calls == 1


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


def test_isolation_requirement_not_weakened_by_missing_task_flag() -> None:
    registry = ToolRegistry()
    register_sandbox_tools(registry, ToolWiringContext())
    contract = registry.get("script.run").contract
    availability = SandboxIsolationAvailability(
        session_configured=False,
        host_configured=False,
    )
    with pytest.raises(SandboxIsolationRequiredError):
        require_sandbox_isolation(
            contract=contract,
            availability=availability,
            run_id=_RUN_ID,
            agent_id=_AGENT_ID,
        )
