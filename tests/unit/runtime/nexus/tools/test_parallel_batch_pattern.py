# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-9 — ParallelBatchPattern acceptance tests."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.config_types import ToolInvocationMode
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.contracts.execution_identity import TaskId
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.patterns.parallel_batch import ParallelBatchPattern
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.nexus.tools.tool_invocation_pattern import pattern_for_mode
from intergrax.runtime.nexus.tools.tool_loop import execute_planned_tool_calls
from intergrax.tools.core.tool_plan import PlannedToolCall, ToolCallPlan
from intergrax.tools.core.tool_plan_decision import ToolPlanDecision
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.registry import ToolRegistry
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager, canonical_execution_identity_scope, canonical_run_id_for_tests, tools_agent_make_contract

pytestmark = pytest.mark.unit

_SLEEP_S = 0.08


class _In(BaseModel):
    value: int = 1


class _Out(BaseModel):
    result: int = 0


class _ConcurrencyTracker:
    active: int = 0
    max_active: int = 0
    lock: threading.Lock = threading.Lock()


class _SlowReadOnlyHandler:
    def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
        with _ConcurrencyTracker.lock:
            _ConcurrencyTracker.active += 1
            _ConcurrencyTracker.max_active = max(_ConcurrencyTracker.max_active, _ConcurrencyTracker.active)
        time.sleep(_SLEEP_S)
        with _ConcurrencyTracker.lock:
            _ConcurrencyTracker.active -= 1
        return _Out(result=request.input.value)


class _SlowMutatingHandler:
    def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
        time.sleep(_SLEEP_S)
        return _Out(result=request.input.value * 10)


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool_id in ("read.a", "read.b", "read.c"):
        registry.register(
            tools_agent_make_contract(tool_id, _In, _Out),
            _SlowReadOnlyHandler(),
        )
    mutating = ToolContract(
        tool_id="write.mutate",
        name="write.mutate",
        description="mutating tool",
        input_schema=_In,
        output_schema=_Out,
        error_mapping={},
        side_effects=True,
    )
    registry.register(mutating, _SlowMutatingHandler())
    return registry


def _runtime_state(registry: ToolRegistry) -> RuntimeState:
    run_id = canonical_run_id_for_tests("run-parallel")
    task_id = TaskId(f"task_{run_id[4:]}")
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        max_parallel_tool_calls=3,
        tool_invocation_mode=ToolInvocationMode.PARALLEL_BATCH,
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
            message="parallel probe",
            task_id=task_id,
            run_id=run_id,
        ),
        run_id=run_id,
    )


def test_pattern_for_mode_parallel_batch() -> None:
    pattern = pattern_for_mode(ToolInvocationMode.PARALLEL_BATCH)
    assert pattern.pattern_id == "parallel_batch"


def test_parallel_read_only_faster_than_serial() -> None:
    registry = _registry()
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    state = _runtime_state(registry)
    calls = [
        PlannedToolCall(step_id=f"s{i}", tool_id=tool_id, input=_In(value=i))
        for i, tool_id in enumerate(("read.a", "read.b", "read.c"))
    ]

    _ConcurrencyTracker.active = 0
    _ConcurrencyTracker.max_active = 0
    serial_start = time.perf_counter()
    with canonical_execution_identity_scope(state.run_id):
        execute_planned_tool_calls(
            state=state,
            invoker=invoker,
            calls=calls,
            idempotency_prefix="serial",
            max_parallel_read_only=1,
        )
    serial_elapsed = time.perf_counter() - serial_start
    serial_peak = _ConcurrencyTracker.max_active

    _ConcurrencyTracker.active = 0
    _ConcurrencyTracker.max_active = 0
    parallel_start = time.perf_counter()
    with canonical_execution_identity_scope(state.run_id):
        execute_planned_tool_calls(
            state=state,
            invoker=invoker,
            calls=calls,
            idempotency_prefix="parallel",
            max_parallel_read_only=3,
        )
    parallel_elapsed = time.perf_counter() - parallel_start
    parallel_peak = _ConcurrencyTracker.max_active

    assert serial_peak == 1
    assert parallel_peak >= 2
    assert parallel_elapsed < serial_elapsed * 0.85


def test_mutating_calls_stay_serial_after_read_only_batch() -> None:
    registry = _registry()
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    state = _runtime_state(registry)
    calls = [
        PlannedToolCall(step_id="r1", tool_id="read.a", input=_In(value=1)),
        PlannedToolCall(step_id="w1", tool_id="write.mutate", input=_In(value=2)),
        PlannedToolCall(step_id="r2", tool_id="read.b", input=_In(value=3)),
    ]
    with canonical_execution_identity_scope(state.run_id):
        outcomes = execute_planned_tool_calls(
            state=state,
            invoker=invoker,
            calls=calls,
            idempotency_prefix="mixed",
            max_parallel_read_only=3,
        )
    traces = [outcome.trace for outcome in outcomes]
    assert [trace.tool_name for trace in traces] == ["read.a", "write.mutate", "read.b"]
    assert traces[1].output_preview == '{"result":20}'


class _Planner:
    def plan_tools(self, input_data, context=None, *, run_id, allowed_tool_ids=None):
        _ = input_data, context, run_id, allowed_tool_ids
        return ToolPlanDecision(
            final_answer=None,
            tool_plan=ToolCallPlan(
                calls=[
                    PlannedToolCall(step_id="s1", tool_id="read.a", input=_In(value=1)),
                    PlannedToolCall(step_id="s2", tool_id="read.b", input=_In(value=2)),
                ]
            ),
            messages=[],
        )


def test_parallel_batch_pattern_returns_aggregate() -> None:
    registry = _registry()
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    state = _runtime_state(registry)
    with canonical_execution_identity_scope(state.run_id):
        result = ParallelBatchPattern().execute(
            state=state,
            invoker=invoker,
            planner=_Planner(),
            plan=None,
            allowed_tool_ids=None,
            max_iterations=1,
            planner_input="go",
        )
    assert result.aggregate is not None
    assert result.aggregate.success_count == 2
    assert "read.a" in result.aggregate.combined_context
