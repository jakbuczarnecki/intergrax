# © Artur Czarnecki. All rights reserved.

"""UE-6B — canonical ExecutionId across agentic tool-loop mechanics."""

from __future__ import annotations

import ast
import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from intergrax.contracts.agent_execution_result import (
    AgentExecutionResult,
    AgentExecutionStatus,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    peek_active_execution_id,
    peek_active_execution_identity,
    require_active_execution_id,
    require_active_execution_identity,
)
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.nexus.tools.tool_loop import (
    execute_planned_tool_calls,
    run_bounded_tool_loop,
)
from intergrax.runtime.nexus.tools.tool_planner_protocol import IterativeToolPlannerProtocol
from intergrax.tools.core.tool_plan import PlannedToolCall, ToolCallPlan
from intergrax.tools.core.tool_plan_decision import ToolPlanDecision
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult
from intergrax.tools.registry import ToolRegistry
from testing_support.builder import (
    build_in_memory_session_manager,
    build_runtime_state_for_tests,
    tools_agent_make_contract,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_FORBIDDEN_TOOL_LOOP_TOKENS = frozenset(
    {
        "mint_execution_id",
        "mint_attempt_id",
        "mint_run_id",
        "transition_active_execution_identity",
        "bind_active_execution_identity",
        "reset_active_execution_identity",
        "ExecutionBoundary",
        "ExecutionIdentityBinding",
    }
)

_FORBIDDEN_DYNAMIC_TOKENS = frozenset(
    {
        "Any",
        "getattr",
        "setattr",
        "hasattr",
        "__getattr__",
        "inspect",
        "importlib",
        "type: ignore",
    }
)


@dataclass(frozen=True, slots=True)
class IdentityObservation:
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId


class _In(BaseModel):
    value: int = 1


class _Out(BaseModel):
    result: int = 0


class _Handler:
    def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
        return _Out(result=request.input.value)


class _FailingHandler:
    def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
        _ = request
        raise RuntimeError("tool-side-effect-failure")


class _SlowReadOnlyHandler:
    def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
        time.sleep(0.05)
        return _Out(result=request.input.value)


class RecordingIdentityToolInvoker(RuntimeToolInvoker):
    def __init__(
        self,
        *,
        registry: ToolRegistry,
        executor: RegistryToolExecutor,
    ) -> None:
        super().__init__(registry=registry, executor=executor)
        self.observations: list[IdentityObservation] = []

    def invoke(
        self,
        state: RuntimeState,
        request: ToolExecutionRequest[BaseModel],
        agent_id: str,
    ) -> ToolExecutionResult[BaseModel]:
        run_id, attempt_id = require_active_execution_identity()
        execution_id = require_active_execution_id()
        self.observations.append(
            IdentityObservation(
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            )
        )
        return super().invoke(state=state, request=request, agent_id=agent_id)


def _identity_binding(*, run_id: RunId | None = None) -> ExecutionIdentityBinding:
    return ExecutionIdentityBinding(
        run_id=run_id or mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _bind_canonical_execution_context(identity: ExecutionIdentityBinding) -> tuple[object, object]:
    from intergrax.contracts.execution_identity import (
        bind_active_execution_identity,
    )
    from intergrax.runtime.execution.active_execution_budget import bind_root_execution_budget
    from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger

    identity_token = bind_active_execution_identity(
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
        execution_id=identity.execution_id,
    )
    budget_token = bind_root_execution_budget(
        execution_id=identity.execution_id,
        ledger=create_execution_budget_ledger(None),
    )
    return identity_token, budget_token


def _reset_canonical_execution_context(identity_token: object, budget_token: object) -> None:
    from intergrax.contracts.execution_identity import reset_active_execution_identity
    from intergrax.runtime.execution.active_execution_budget import reset_active_execution_budget

    reset_active_execution_budget(budget_token)
    reset_active_execution_identity(identity_token)


def _runtime_state(*, run_id: RunId) -> RuntimeState:
    config = RuntimeConfig(
        llm_adapter=None,
        production_mode=False,
        max_parallel_tool_calls=3,
        max_tool_iterations=2,
    )
    ctx = RuntimeContext(
        config=config,
        session_manager=build_in_memory_session_manager(),
        prompt_registry=MagicMock(),
    )
    return RuntimeState(
        context=ctx,
        request=RuntimeRequest(
            agent_id="agent-1",
            user_id="user-1",
            session_id="session-1",
            tenant_id="tenant-1",
            message="tool identity probe",
            task_id=mint_task_id(),
            run_id=run_id,
        ),
        run_id=run_id,
    )


def _registry(*, handler: object | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("probe.read", _In, _Out),
        handler or _Handler(),
    )
    return registry


def _recording_invoker(
    registry: ToolRegistry | None = None,
    *,
    handler: object | None = None,
) -> RecordingIdentityToolInvoker:
    reg = registry or _registry(handler=handler)
    return RecordingIdentityToolInvoker(
        registry=reg,
        executor=RegistryToolExecutor(reg),
    )


class _SinglePlanner:
    def plan_tools(self, input_data, context=None, *, run_id, allowed_tool_ids=None, tool_choice=None):
        _ = input_data, context, run_id, allowed_tool_ids, tool_choice
        return ToolPlanDecision(
            final_answer=None,
            tool_plan=ToolCallPlan(
                calls=[
                    PlannedToolCall(
                        step_id="step-1",
                        tool_id="probe.read",
                        input=_In(value=3),
                    )
                ]
            ),
            messages=[],
        )


def _decision_note(*basis_ids: str, purpose: str) -> str:
    basis = ",".join(basis_ids)
    return f"EVIDENCE_BASIS: {basis}\nPURPOSE: {purpose}"


class _TwoRoundPlanner:
    def __init__(self) -> None:
        self._round = 0

    def plan_tools(self, input_data, context=None, *, run_id, allowed_tool_ids=None):
        _ = input_data, context, run_id, allowed_tool_ids
        return ToolPlanDecision(final_answer=None, tool_plan=None, messages=[])

    def plan_native_round(
        self,
        messages: list[ChatMessage],
        *,
        allowed_tool_ids=None,
        run_id: str,
        tool_choice=None,
    ) -> tuple[LLMAdapterResponse, ToolCallPlan]:
        _ = messages, allowed_tool_ids, run_id, tool_choice
        self._round += 1
        if self._round == 1:
            return (
                LLMAdapterResponse(
                    content=_decision_note("basis-1", purpose="probe round one"),
                    tool_calls=(
                        LLMToolCall.from_openai_shape(
                            call_id="tc-1",
                            name="probe.read",
                            arguments={"value": 1},
                        ),
                    ),
                ),
                ToolCallPlan(
                    calls=[
                        PlannedToolCall(
                            step_id="step-1",
                            tool_id="probe.read",
                            input=_In(value=1),
                        )
                    ]
                ),
            )
        return (
            LLMAdapterResponse(
                content=_decision_note("tc-1", purpose="probe round two"),
                tool_calls=(
                    LLMToolCall.from_openai_shape(
                        call_id="tc-2",
                        name="probe.read",
                        arguments={"value": 2},
                    ),
                ),
            ),
            ToolCallPlan(
                calls=[
                    PlannedToolCall(
                        step_id="step-2",
                        tool_id="probe.read",
                        input=_In(value=2),
                    )
                ]
            ),
        )


class _ParallelReadOnlyPlanner:
    def plan_tools(self, input_data, context=None, *, run_id, allowed_tool_ids=None, tool_choice=None):
        _ = input_data, context, run_id, allowed_tool_ids, tool_choice
        return ToolPlanDecision(
            final_answer=None,
            tool_plan=ToolCallPlan(
                calls=[
                    PlannedToolCall(
                        step_id=f"step-{index}",
                        tool_id="probe.read",
                        input=_In(value=index),
                    )
                    for index in range(3)
                ]
            ),
            messages=[],
        )


class ToolLoopExecutionDelegate:
    def __init__(
        self,
        *,
        invoker: RecordingIdentityToolInvoker,
        planner: object,
        max_iterations: int = 1,
        max_parallel_read_only: int = 1,
    ) -> None:
        self._invoker = invoker
        self._planner = planner
        self._max_iterations = max_iterations
        self._max_parallel_read_only = max_parallel_read_only

    async def execute(self, request: RuntimeRequest) -> AgentExecutionResult:
        from intergrax.runtime.execution.active_execution_budget import (
            bind_root_execution_budget,
            peek_active_execution_budget,
            reset_active_execution_budget,
        )
        from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
        from intergrax.contracts.execution_identity import require_active_execution_id

        budget_token = None
        if peek_active_execution_budget() is None:
            budget_token = bind_root_execution_budget(
                execution_id=require_active_execution_id(),
                ledger=create_execution_budget_ledger(None),
            )
        state = _runtime_state(run_id=request.run_id)
        state.context.config.max_parallel_tool_calls = self._max_parallel_read_only
        run_bounded_tool_loop(
            state=state,
            invoker=self._invoker,
            tool_planner=self._planner,
            planner_input=[ChatMessage(role="user", content=request.message)],
            allowed_tool_ids=("probe.read",),
            max_iterations=self._max_iterations,
        )
        if budget_token is not None:
            reset_active_execution_budget(budget_token)
        return AgentExecutionResult(
            agent_id=request.agent_id,
            run_id=request.run_id,
            status=AgentExecutionStatus.COMPLETED,
            summary="tool-loop-complete",
        )


def test_sequential_tool_calls_observe_active_identity() -> None:
    identity = _identity_binding()
    state = _runtime_state(run_id=identity.run_id)
    invoker = _recording_invoker()
    calls = [
        PlannedToolCall(step_id="s1", tool_id="probe.read", input=_In(value=1)),
        PlannedToolCall(step_id="s2", tool_id="probe.read", input=_In(value=2)),
    ]

    token, budget_token = _bind_canonical_execution_context(identity)
    try:
        execute_planned_tool_calls(
            state=state,
            invoker=invoker,
            calls=calls,
            idempotency_prefix="seq",
            max_parallel_read_only=1,
        )
    finally:
        _reset_canonical_execution_context(token, budget_token)

    assert len(invoker.observations) == 2
    for observation in invoker.observations:
        assert observation.run_id == identity.run_id
        assert observation.attempt_id == identity.attempt_id
        assert observation.execution_id == identity.execution_id


def test_bounded_react_iterations_preserve_execution_id() -> None:
    identity = _identity_binding()
    state = _runtime_state(run_id=identity.run_id)
    invoker = _recording_invoker()
    planner = _TwoRoundPlanner()

    token, budget_token = _bind_canonical_execution_context(identity)
    try:
        result = run_bounded_tool_loop(
            state=state,
            invoker=invoker,
            tool_planner=planner,
            planner_input=[ChatMessage(role="user", content="iterate")],
            allowed_tool_ids=("probe.read",),
            max_iterations=2,
        )
    finally:
        _reset_canonical_execution_context(token, budget_token)

    assert result.loop_iterations == 2
    assert len(invoker.observations) == 2
    assert all(obs.execution_id == identity.execution_id for obs in invoker.observations)
    assert all(obs.attempt_id == identity.attempt_id for obs in invoker.observations)


def test_no_active_identity_fails_before_tool_side_effect() -> None:
    handler_calls = 0

    class _CountingHandler:
        def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
            nonlocal handler_calls
            handler_calls += 1
            return _Out(result=request.input.value)

    state = build_runtime_state_for_tests(run_id="run-no-identity")
    invoker = _recording_invoker(handler=_CountingHandler())
    calls = [PlannedToolCall(step_id="s1", tool_id="probe.read", input=_In(value=1))]

    with pytest.raises(RuntimeError, match="active execution identity required"):
        execute_planned_tool_calls(
            state=state,
            invoker=invoker,
            calls=calls,
            idempotency_prefix="no-id",
        )

    assert handler_calls == 0
    assert invoker.observations == []


def test_run_id_mismatch_fails_before_invoke() -> None:
    identity = _identity_binding()
    mismatched_run = mint_run_id()
    state = _runtime_state(run_id=mismatched_run)
    invoker = _recording_invoker()
    calls = [PlannedToolCall(step_id="s1", tool_id="probe.read", input=_In(value=1))]

    token, budget_token = _bind_canonical_execution_context(identity)
    try:
        with pytest.raises(
            RuntimeError,
            match="tool execution run_id does not match active execution",
        ):
            execute_planned_tool_calls(
                state=state,
                invoker=invoker,
                calls=calls,
                idempotency_prefix="mismatch",
            )
    finally:
        _reset_canonical_execution_context(token, budget_token)

    assert invoker.observations == []


def test_parallel_read_only_workers_share_execution_id() -> None:
    identity = _identity_binding()
    state = _runtime_state(run_id=identity.run_id)
    invoker = _recording_invoker(handler=_SlowReadOnlyHandler())
    calls = [
        PlannedToolCall(step_id=f"s{index}", tool_id="probe.read", input=_In(value=index))
        for index in range(3)
    ]

    token, budget_token = _bind_canonical_execution_context(identity)
    try:
        execute_planned_tool_calls(
            state=state,
            invoker=invoker,
            calls=calls,
            idempotency_prefix="parallel",
            max_parallel_read_only=3,
        )
    finally:
        _reset_canonical_execution_context(token, budget_token)

    assert len(invoker.observations) == 3
    execution_ids = {obs.execution_id for obs in invoker.observations}
    attempt_ids = {obs.attempt_id for obs in invoker.observations}
    run_ids = {obs.run_id for obs in invoker.observations}
    assert execution_ids == {identity.execution_id}
    assert attempt_ids == {identity.attempt_id}
    assert run_ids == {identity.run_id}


@pytest.mark.asyncio
async def test_execution_boundary_tool_loop_matches_boundary_identity() -> None:
    identity = _identity_binding()
    invoker = _recording_invoker()
    delegate = ToolLoopExecutionDelegate(invoker=invoker, planner=_SinglePlanner())
    boundary = ExecutionBoundary(delegate, identity=identity)
    request = RuntimeRequest(
        agent_id="agent-1",
        user_id="user-1",
        session_id="session-1",
        tenant_id="tenant-1",
        message="boundary probe",
        task_id=mint_task_id(),
        run_id=identity.run_id,
    )

    result = await boundary.execute(request)

    assert result.status is AgentExecutionStatus.COMPLETED
    assert len(invoker.observations) == 1


@pytest.mark.asyncio
async def test_concurrent_executions_do_not_leak_identity() -> None:
    identity_a = _identity_binding()
    identity_b = _identity_binding()
    invoker_a = _recording_invoker()
    invoker_b = _recording_invoker()

    async def _run(identity: ExecutionIdentityBinding, invoker: RecordingIdentityToolInvoker) -> None:
        delegate = ToolLoopExecutionDelegate(
            invoker=invoker,
            planner=_ParallelReadOnlyPlanner(),
            max_parallel_read_only=3,
        )
        boundary = ExecutionBoundary(delegate, identity=identity)
        request = RuntimeRequest(
            agent_id="agent-1",
            user_id="user-1",
            session_id="session-1",
            tenant_id="tenant-1",
            message="concurrent probe",
            task_id=mint_task_id(),
            run_id=identity.run_id,
        )
        await boundary.execute(request)

    await asyncio.gather(
        _run(identity_a, invoker_a),
        _run(identity_b, invoker_b),
    )

    assert all(obs.execution_id == identity_a.execution_id for obs in invoker_a.observations)
    assert all(obs.execution_id == identity_b.execution_id for obs in invoker_b.observations)
    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None


@pytest.mark.asyncio
async def test_tool_failure_preserves_identity_until_boundary_reset() -> None:
    identity = _identity_binding()
    invoker = _recording_invoker(handler=_FailingHandler())
    delegate = ToolLoopExecutionDelegate(invoker=invoker, planner=_SinglePlanner())
    boundary = ExecutionBoundary(delegate, identity=identity)
    request = RuntimeRequest(
        agent_id="agent-1",
        user_id="user-1",
        session_id="session-1",
        tenant_id="tenant-1",
        message="failure probe",
        task_id=mint_task_id(),
        run_id=identity.run_id,
    )

    result = await boundary.execute(request)

    assert result.status is AgentExecutionStatus.COMPLETED
    assert len(invoker.observations) == 1
    assert invoker.observations[0].execution_id == identity.execution_id
    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None


def test_worker_thread_pool_reuse_does_not_retain_copied_context() -> None:
    identity = _identity_binding()
    state = _runtime_state(run_id=identity.run_id)
    invoker = _recording_invoker(handler=_SlowReadOnlyHandler())
    calls = [
        PlannedToolCall(step_id=f"s{index}", tool_id="probe.read", input=_In(value=index))
        for index in range(3)
    ]

    token, budget_token = _bind_canonical_execution_context(identity)
    try:
        execute_planned_tool_calls(
            state=state,
            invoker=invoker,
            calls=calls,
            idempotency_prefix="reuse",
            max_parallel_read_only=3,
        )
        execute_planned_tool_calls(
            state=state,
            invoker=invoker,
            calls=calls,
            idempotency_prefix="reuse-2",
            max_parallel_read_only=3,
        )
    finally:
        _reset_canonical_execution_context(token, budget_token)

    assert all(obs.execution_id == identity.execution_id for obs in invoker.observations)


def test_two_round_planner_is_iterative_protocol() -> None:
    assert isinstance(_TwoRoundPlanner(), IterativeToolPlannerProtocol)


def test_tool_loop_module_has_no_forbidden_identity_tokens() -> None:
    source = Path("intergrax/runtime/nexus/tools/tool_loop.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    tokens: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            tokens.add(node.id)
        elif isinstance(node, ast.Attribute):
            tokens.add(node.attr)

    forbidden = tokens & _FORBIDDEN_TOOL_LOOP_TOKENS
    assert not forbidden, f"forbidden identity tokens in tool_loop.py: {sorted(forbidden)}"
    assert "copy_context" in tokens
    assert "require_active_execution_id" in tokens
    assert "require_active_execution_identity" in tokens


def test_tool_loop_module_has_no_forbidden_dynamic_tokens() -> None:
    source = Path("intergrax/runtime/nexus/tools/tool_loop.py").read_text(encoding="utf-8")
    for token in _FORBIDDEN_DYNAMIC_TOKENS:
        assert token not in source
