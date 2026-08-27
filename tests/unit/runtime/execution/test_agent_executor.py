# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.contracts.agent_execution_result import (
    AgentExecutionResult,
    AgentExecutionStatus,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    peek_active_execution_id,
    peek_active_execution_identity,
    require_active_execution_id,
    require_active_execution_identity,
    reset_active_execution_identity,
)
from intergrax.runtime.execution import (
    ExecutionCapability,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
)
from intergrax.runtime.execution.boundary import (
    ExecutionAdmissionHook,
    ExecutionBoundary,
    ExecutionIdentityBinding,
)
from intergrax.runtime.execution.facade import Execution
from intergrax.runtime.execution.agentic import AgentExecutor
from intergrax.runtime.execution.strategy import ExecutionStrategy, StrategyResolver
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_FORBIDDEN_AGENTIC_TOKENS = frozenset(
    {
        "GraphExecutor",
        "NexusLoop",
        "UAEPExecutor",
        "ToolRuntime",
        "Task",
        "TaskResult",
        "PolicyEngine",
        "RuntimeEventBus",
        "Checkpoint",
        "ContextManager",
        "AgentRouter",
        "run_agent_with_result",
    }
)

_FORBIDDEN_DYNAMIC_TOKENS = frozenset(
    {
        "Any",
        "dict[",
        "Mapping[",
        "MutableMapping[",
        "getattr",
        "setattr",
        "hasattr",
        "__getattr__",
        "__dict__",
        "vars(",
        "inspect",
        "importlib",
        "isinstance(",
        "issubclass(",
        "callable(",
        "**kwargs",
    }
)


@dataclass(frozen=True, slots=True)
class OtherOutputType:
    value: str


class RecordingAgentEngine:
    """Typed fake satisfying AgentEnginePort without MagicMock."""

    def __init__(
        self,
        *,
        result: AgentExecutionResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.calls: list[RuntimeRequest] = []
        self.observed_run_id: RunId | None = None
        self.observed_execution_id: ExecutionId | None = None

    async def run_with_result(
        self,
        request: RuntimeRequest,
    ) -> AgentExecutionResult:
        self.calls.append(request)
        run_id, _attempt_id = require_active_execution_identity()
        self.observed_run_id = run_id
        self.observed_execution_id = require_active_execution_id()

        if self._error is not None:
            raise self._error

        if self._result is not None:
            return self._result

        return AgentExecutionResult(
            agent_id=request.agent_id,
            run_id=request.run_id,
            status=AgentExecutionStatus.COMPLETED,
            summary="recorded",
        )


class IdentityProbingAdmissionHook:
    def __init__(self, captured: dict[str, RunId | AttemptId | ExecutionId]) -> None:
        self._captured = captured
        self.admit_count = 0

    async def admit(
        self,
        request: ExecutionRequest[RuntimeRequest, AgentExecutionResult],
    ) -> None:
        self.admit_count += 1
        run_id, attempt_id = require_active_execution_identity()
        execution_id = require_active_execution_id()
        self._captured["hook_run_id"] = run_id
        self._captured["hook_attempt_id"] = attempt_id
        self._captured["hook_execution_id"] = execution_id
        self._captured["hook_runtime_run_id"] = request.input.run_id


def _runtime_request(
    *,
    run_id: RunId | None = None,
    agent_id: str = "test-agent",
) -> RuntimeRequest:
    return RuntimeRequest(
        agent_id=agent_id,
        user_id="user-1",
        session_id="session-1",
        message="hello",
        task_id=mint_task_id(),
        run_id=run_id or mint_run_id(),
    )


def _agentic_request(
    runtime_request: RuntimeRequest,
    *,
    capabilities: frozenset[ExecutionCapability] = frozenset(
        {ExecutionCapability.TOOLS}
    ),
    output_type: type[AgentExecutionResult] | None = AgentExecutionResult,
) -> ExecutionRequest[RuntimeRequest, AgentExecutionResult]:
    return ExecutionRequest(
        input=runtime_request,
        output_type=output_type,
        capabilities=capabilities,
    )


def _identity_binding(*, run_id: RunId | None = None) -> ExecutionIdentityBinding:
    return ExecutionIdentityBinding(
        run_id=run_id or mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _agentic_stack(
    engine: RecordingAgentEngine,
    *,
    identity: ExecutionIdentityBinding | None = None,
    admission_hooks: tuple[ExecutionAdmissionHook, ...] = (),
) -> Execution[
    ExecutionRequest[RuntimeRequest, AgentExecutionResult],
    ExecutionResult[AgentExecutionResult],
]:
    executor = AgentExecutor(engine)
    boundary = ExecutionBoundary(
        executor,
        admission_hooks=admission_hooks,
        identity=identity,
    )
    return Execution(boundary)


@pytest.mark.asyncio
async def test_tools_capabilities_resolve_to_agentic_and_execute_full_path() -> None:
    identity = _identity_binding()
    runtime_request = _runtime_request(run_id=identity.run_id)
    request = _agentic_request(runtime_request)
    engine = RecordingAgentEngine()
    captured: dict[str, RunId | AttemptId | ExecutionId] = {}
    admission_hook = IdentityProbingAdmissionHook(captured)
    execution = _agentic_stack(
        engine,
        identity=identity,
        admission_hooks=(admission_hook,),
    )

    assert StrategyResolver().resolve(request) is ExecutionStrategy.AGENTIC

    result = await execution.execute(request)

    assert result.status is ExecutionStatus.COMPLETED
    assert result.output.status is AgentExecutionStatus.COMPLETED
    assert result.output.run_id == identity.run_id
    assert len(engine.calls) == 1
    assert engine.calls[0] is runtime_request
    assert engine.observed_run_id == identity.run_id
    assert engine.observed_execution_id == identity.execution_id
    assert admission_hook.admit_count == 1
    assert captured["hook_execution_id"] == identity.execution_id
    assert captured["hook_runtime_run_id"] == identity.run_id
    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None


@pytest.mark.asyncio
async def test_runtime_request_run_id_mismatch_fails_before_engine() -> None:
    identity = _identity_binding()
    runtime_request = _runtime_request(run_id=mint_run_id())
    engine = RecordingAgentEngine()
    execution = _agentic_stack(engine, identity=identity)

    with pytest.raises(
        RuntimeError,
        match="agentic RuntimeRequest run_id does not match active execution",
    ):
        await execution.execute(_agentic_request(runtime_request))

    assert engine.calls == []


@pytest.mark.asyncio
async def test_agent_executor_requires_active_execution_identity() -> None:
    engine = RecordingAgentEngine()
    executor = AgentExecutor(engine)
    runtime_request = _runtime_request()

    with pytest.raises(RuntimeError, match="active execution identity required"):
        await executor.execute(_agentic_request(runtime_request))

    assert engine.calls == []


@pytest.mark.asyncio
async def test_agent_executor_requires_active_execution_id() -> None:
    engine = RecordingAgentEngine()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_id)
    runtime_request = _runtime_request(run_id=run_id)
    try:
        executor = AgentExecutor(engine)
        with pytest.raises(RuntimeError, match="active ExecutionId required"):
            await executor.execute(_agentic_request(runtime_request))
    finally:
        reset_active_execution_identity(token)

    assert engine.calls == []


@pytest.mark.asyncio
async def test_identity_resets_after_success() -> None:
    identity = _identity_binding()
    runtime_request = _runtime_request(run_id=identity.run_id)
    execution = _agentic_stack(RecordingAgentEngine(), identity=identity)

    await execution.execute(_agentic_request(runtime_request))

    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None


@pytest.mark.asyncio
async def test_identity_resets_after_engine_exception() -> None:
    identity = _identity_binding()
    runtime_request = _runtime_request(run_id=identity.run_id)
    engine_error = ValueError("engine-failure")
    execution = _agentic_stack(
        RecordingAgentEngine(error=engine_error),
        identity=identity,
    )

    with pytest.raises(ValueError) as exc_info:
        await execution.execute(_agentic_request(runtime_request))

    assert exc_info.value is engine_error
    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None


@pytest.mark.asyncio
async def test_empty_capabilities_rejected_before_engine() -> None:
    identity = _identity_binding()
    runtime_request = _runtime_request(run_id=identity.run_id)
    engine = RecordingAgentEngine()
    execution = _agentic_stack(engine, identity=identity)

    with pytest.raises(RuntimeError, match="AgentExecutor requires AGENTIC strategy"):
        await execution.execute(_agentic_request(runtime_request, capabilities=frozenset()))

    assert engine.calls == []


@pytest.mark.asyncio
async def test_orchestration_rejected_before_engine() -> None:
    identity = _identity_binding()
    runtime_request = _runtime_request(run_id=identity.run_id)
    engine = RecordingAgentEngine()
    execution = _agentic_stack(engine, identity=identity)

    with pytest.raises(RuntimeError, match="AgentExecutor requires AGENTIC strategy"):
        await execution.execute(
            _agentic_request(
                runtime_request,
                capabilities=frozenset({ExecutionCapability.ORCHESTRATION}),
            )
        )

    assert engine.calls == []


@pytest.mark.asyncio
async def test_tools_and_orchestration_rejected_before_engine() -> None:
    identity = _identity_binding()
    runtime_request = _runtime_request(run_id=identity.run_id)
    engine = RecordingAgentEngine()
    execution = _agentic_stack(engine, identity=identity)

    with pytest.raises(RuntimeError, match="AgentExecutor requires AGENTIC strategy"):
        await execution.execute(
            _agentic_request(
                runtime_request,
                capabilities=frozenset(
                    {
                        ExecutionCapability.TOOLS,
                        ExecutionCapability.ORCHESTRATION,
                    }
                ),
            )
        )

    assert engine.calls == []


@pytest.mark.asyncio
async def test_tools_and_streaming_rejected_before_engine() -> None:
    identity = _identity_binding()
    runtime_request = _runtime_request(run_id=identity.run_id)
    engine = RecordingAgentEngine()
    execution = _agentic_stack(engine, identity=identity)

    with pytest.raises(RuntimeError, match="agentic streaming is not implemented"):
        await execution.execute(
            _agentic_request(
                runtime_request,
                capabilities=frozenset(
                    {
                        ExecutionCapability.TOOLS,
                        ExecutionCapability.STREAMING,
                    }
                ),
            )
        )

    assert engine.calls == []


@pytest.mark.asyncio
async def test_output_type_agent_execution_result_accepted() -> None:
    identity = _identity_binding()
    runtime_request = _runtime_request(run_id=identity.run_id)
    execution = _agentic_stack(RecordingAgentEngine(), identity=identity)

    result = await execution.execute(_agentic_request(runtime_request))

    assert result.output is not None


@pytest.mark.asyncio
async def test_output_type_none_rejected_before_engine() -> None:
    identity = _identity_binding()
    runtime_request = _runtime_request(run_id=identity.run_id)
    engine = RecordingAgentEngine()
    execution = _agentic_stack(engine, identity=identity)

    with pytest.raises(
        RuntimeError,
        match="AgentExecutor requires AgentExecutionResult output_type",
    ):
        await execution.execute(_agentic_request(runtime_request, output_type=None))

    assert engine.calls == []


@pytest.mark.asyncio
async def test_other_output_type_rejected_before_engine() -> None:
    identity = _identity_binding()
    runtime_request = _runtime_request(run_id=identity.run_id)
    engine = RecordingAgentEngine()
    execution = _agentic_stack(engine, identity=identity)
    request = ExecutionRequest(
        input=runtime_request,
        output_type=OtherOutputType,
        capabilities=frozenset({ExecutionCapability.TOOLS}),
    )

    with pytest.raises(
        RuntimeError,
        match="AgentExecutor requires AgentExecutionResult output_type",
    ):
        await execution.execute(request)

    assert engine.calls == []


@pytest.mark.asyncio
async def test_inner_failed_agent_result_wraps_as_completed_execution_result() -> None:
    """
    Strategy-native failure normalization is not frozen in UE-6A.

    AgentExecutor returns ExecutionStatus.COMPLETED when AgentEngine returns
    normally, even when inner AgentExecutionResult.status is FAILED.
    """
    identity = _identity_binding()
    runtime_request = _runtime_request(run_id=identity.run_id)
    failed_result = AgentExecutionResult(
        agent_id=runtime_request.agent_id,
        run_id=runtime_request.run_id,
        status=AgentExecutionStatus.FAILED,
        summary="",
        errors=["blocked"],
    )
    execution = _agentic_stack(
        RecordingAgentEngine(result=failed_result),
        identity=identity,
    )

    result = await execution.execute(_agentic_request(runtime_request))

    assert result.status is ExecutionStatus.COMPLETED
    assert result.output is failed_result
    assert result.output.status is AgentExecutionStatus.FAILED


@pytest.mark.asyncio
async def test_engine_exception_propagates_unchanged() -> None:
    identity = _identity_binding()
    runtime_request = _runtime_request(run_id=identity.run_id)
    engine_error = RuntimeError("uaep-bypass")
    engine = RecordingAgentEngine(error=engine_error)
    execution = _agentic_stack(engine, identity=identity)

    with pytest.raises(RuntimeError) as exc_info:
        await execution.execute(_agentic_request(runtime_request))

    assert exc_info.value is engine_error


def test_agentic_module_has_no_forbidden_import_tokens() -> None:
    agentic_path = Path("intergrax/runtime/execution/agentic.py")
    source = agentic_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)

    for forbidden in (
        "intergrax.agents",
        "intergrax.runtime.nexus.execution",
        "intergrax.runtime.nexus.orchestration",
    ):
        assert not any(
            name == forbidden or name.startswith(f"{forbidden}.") for name in imported
        )


def test_agentic_source_has_no_forbidden_tokens() -> None:
    source = Path("intergrax/runtime/execution/agentic.py").read_text(encoding="utf-8")
    for token in _FORBIDDEN_AGENTIC_TOKENS:
        assert token not in source, f"forbidden token in agentic.py: {token}"


def test_agentic_source_has_no_forbidden_dynamic_mechanisms() -> None:
    source = Path("intergrax/runtime/execution/agentic.py").read_text(encoding="utf-8")
    for token in _FORBIDDEN_DYNAMIC_TOKENS:
        assert token not in source, f"forbidden dynamic token in agentic.py: {token}"


def test_agent_executor_not_exported_from_package_root() -> None:
    import intergrax.runtime.execution as execution_package

    assert "AgentExecutor" not in execution_package.__all__
    assert "AgentEnginePort" not in execution_package.__all__


def test_agent_executor_does_not_bind_or_reset_identity() -> None:
    source = Path("intergrax/runtime/execution/agentic.py").read_text(encoding="utf-8")
    assert "bind_active_execution_identity" not in source
    assert "reset_active_execution_identity" not in source
    assert "mint_execution_id" not in source
    assert "mint_run_id" not in source
    assert "mint_attempt_id" not in source
