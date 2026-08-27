# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-16 — ToolInvocationPattern protocol conformance."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.runtime.nexus.config_types import ToolInvocationMode
from intergrax.runtime.nexus.tools.patterns.bounded_react import BoundedReactPattern
from intergrax.runtime.nexus.tools.patterns.single_pass import SinglePassPattern
from intergrax.runtime.nexus.tools.tool_invocation_pattern import pattern_for_mode
from intergrax.runtime.nexus.tools.tool_loop import resolve_tool_invocation_pattern
from intergrax.tools.core.tool_plan import PlannedToolCall, ToolCallPlan
from intergrax.tools.core.tool_plan_decision import ToolPlanDecision
from intergrax.tools.execution_models import ToolExecutionRequest
from testing_support.builder import build_runtime_state_for_tests, canonical_execution_identity_scope, tools_agent_make_contract

pytestmark = pytest.mark.unit


class _In(BaseModel):
    value: int = 1


class _Out(BaseModel):
    result: int = 0


class _Planner:
    def plan_tools(self, input_data, context=None, *, run_id, allowed_tool_ids=None, tool_choice=None):
        _ = tool_choice
        _ = input_data, context, run_id, allowed_tool_ids
        return ToolPlanDecision(
            final_answer=None,
            tool_plan=ToolCallPlan(
                calls=[
                    PlannedToolCall(
                        step_id="step-1",
                        tool_id="demo.tool",
                        input=_In(value=2),
                    )
                ]
            ),
            messages=[],
        )


class _Handler:
    def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
        return _Out(result=request.input.value)


def test_pattern_for_mode_returns_shipped_classes() -> None:
    assert isinstance(pattern_for_mode(ToolInvocationMode.SINGLE_PASS), SinglePassPattern)
    assert isinstance(pattern_for_mode(ToolInvocationMode.BOUNDED_REACT), BoundedReactPattern)


def test_pattern_for_mode_parallel_batch_shipped() -> None:
    from intergrax.runtime.nexus.tools.patterns.parallel_batch import ParallelBatchPattern

    assert isinstance(pattern_for_mode(ToolInvocationMode.PARALLEL_BATCH), ParallelBatchPattern)


def test_resolve_infers_bounded_react_when_max_iterations_gt_one() -> None:
    pattern = resolve_tool_invocation_pattern(invocation_mode=None, max_iterations=3)
    assert isinstance(pattern, BoundedReactPattern)


def test_single_pass_pattern_executes_planned_calls() -> None:
    from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
    from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
    from intergrax.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(tools_agent_make_contract("demo.tool", _In, _Out), _Handler())
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    state = build_runtime_state_for_tests(run_id="run-pattern")

    with canonical_execution_identity_scope(state.run_id):
        result = SinglePassPattern().execute(
            state=state,
            invoker=invoker,
            planner=_Planner(),
            plan=None,
            allowed_tool_ids=("demo.tool",),
            max_iterations=1,
            planner_input="run tool",
        )

    assert result.stop_reason == "legacy_single_pass"
    assert len(result.tool_traces) == 1
    assert result.tool_traces[0].tool_name == "demo.tool"
