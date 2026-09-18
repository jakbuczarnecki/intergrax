# © Artur Czarnecki. All rights reserved.

"""PLUG-02-R1 — public invocation pattern preserves canonical execution evidence."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import pytest
from pydantic import BaseModel

from intergrax.runtime.nexus.errors.tools_required_error import ToolsRequiredError
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.public_tool_invocation_pattern_bridge import (
    PublicToolInvocationPatternBridge,
)
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.nexus.tools.tool_loop import run_bounded_tool_loop
from intergrax.runtime.nexus.tools.tool_planner_protocol import ToolPlannerProtocol
from intergrax.tools.core.tool_plan import ToolCallPlan
from intergrax.tools.core.tool_plan_decision import ToolPlanDecision
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.invocation_pattern.contracts import (
    ToolInvocationInvokerPort,
    ToolInvocationPatternContext,
    ToolInvocationPatternResult,
    ToolInvocationPlannerPort,
)
from intergrax.tools.registry import ToolRegistry
from testing_support.builder import (
    build_runtime_state_for_tests,
    canonical_governed_execution_scope,
    tools_agent_make_contract,
)

pytestmark = pytest.mark.unit


class _In(BaseModel):
    value: int = 1


class _Out(BaseModel):
    result: int = 0


class _FailIn(BaseModel):
    message: str = "fail"


class _FailOut(BaseModel):
    ok: bool = True


class _SideEffectIn(BaseModel):
    token: str = ""


class _SideEffectOut(BaseModel):
    seen: int = 0


_SIDE_EFFECT_COUNTER = {"count": 0}


class _SuccessHandler:
    def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
        return _Out(result=request.input.value)


class _FailHandler:
    def execute(self, request: ToolExecutionRequest[_FailIn]) -> _FailOut:
        raise ValueError(request.input.message)


class _SideEffectHandler:
    def execute(self, request: ToolExecutionRequest[_SideEffectIn]) -> _SideEffectOut:
        _SIDE_EFFECT_COUNTER["count"] += 1
        return _SideEffectOut(seen=_SIDE_EFFECT_COUNTER["count"])


class _NoopPlanner(ToolPlannerProtocol):
    def plan_tools(
        self,
        input_data: str | list[object],
        context: object | None = None,
        *,
        run_id: str,
        allowed_tool_ids: Sequence[str] | None = None,
        tool_choice: object | None = None,
    ) -> ToolPlanDecision:
        _ = input_data, context, run_id, allowed_tool_ids, tool_choice
        return ToolPlanDecision(final_answer=None, tool_plan=None, messages=[])


def _make_invoker() -> RuntimeToolInvoker:
    registry = ToolRegistry()
    registry.register(tools_agent_make_contract("plug.demo", _In, _Out), _SuccessHandler())
    registry.register(
        tools_agent_make_contract("plug.fail", _FailIn, _FailOut),
        _FailHandler(),
    )
    registry.register(
        tools_agent_make_contract("plug.side_effect", _SideEffectIn, _SideEffectOut),
        _SideEffectHandler(),
    )
    return RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))


class _SingleInvokePublicPattern:
    @property
    def pattern_id(self) -> str:
        return "plug_r1_single_invoke"

    def execute(
        self,
        *,
        context: ToolInvocationPatternContext,
        invoker: ToolInvocationInvokerPort,
        planner: ToolInvocationPlannerPort,
        plan: ToolCallPlan | None,
        allowed_tool_ids: Sequence[str] | None,
        max_iterations: int,
        planner_input: str | list[object],
    ) -> ToolInvocationPatternResult:
        _ = planner, plan, allowed_tool_ids, max_iterations, planner_input
        request = ToolExecutionRequest(
            run_id=context.run_id,
            step_id="public-step-a",
            tool_id="plug.demo",
            input=_In(value=7),
            idempotency_key=f"{context.run_id}:plug.demo:public-step-a",
        )
        agent_id = context.agent_id or "agent"
        result = invoker.invoke_tool(agent_id=agent_id, request=request)
        assert result.success
        return ToolInvocationPatternResult(
            pattern_id=self.pattern_id,
            stop_reason="legacy_single_pass",
            loop_iterations=1,
        )


class _EmptyPublicPattern:
    @property
    def pattern_id(self) -> str:
        return "plug_r1_empty"

    def execute(
        self,
        *,
        context: ToolInvocationPatternContext,
        invoker: ToolInvocationInvokerPort,
        planner: ToolInvocationPlannerPort,
        plan: ToolCallPlan | None,
        allowed_tool_ids: Sequence[str] | None,
        max_iterations: int,
        planner_input: str | list[object],
    ) -> ToolInvocationPatternResult:
        _ = context, invoker, planner, plan, allowed_tool_ids, max_iterations, planner_input
        return ToolInvocationPatternResult(
            pattern_id=self.pattern_id,
            stop_reason="empty_tool_calls",
        )


class _DualInvokePublicPattern:
    @property
    def pattern_id(self) -> str:
        return "plug_r1_dual_invoke"

    def execute(
        self,
        *,
        context: ToolInvocationPatternContext,
        invoker: ToolInvocationInvokerPort,
        planner: ToolInvocationPlannerPort,
        plan: ToolCallPlan | None,
        allowed_tool_ids: Sequence[str] | None,
        max_iterations: int,
        planner_input: str | list[object],
    ) -> ToolInvocationPatternResult:
        _ = planner, plan, allowed_tool_ids, max_iterations, planner_input
        agent_id = context.agent_id or "agent"
        first = ToolExecutionRequest(
            run_id=context.run_id,
            step_id="public-step-1",
            tool_id="plug.demo",
            input=_In(value=1),
        )
        second = ToolExecutionRequest(
            run_id=context.run_id,
            step_id="public-step-2",
            tool_id="plug.demo",
            input=_In(value=2),
        )
        invoker.invoke_tool(agent_id=agent_id, request=first)
        invoker.invoke_tool(agent_id=agent_id, request=second)
        return ToolInvocationPatternResult(
            pattern_id=self.pattern_id,
            stop_reason="legacy_single_pass",
            loop_iterations=1,
        )


class _FailingInvokePublicPattern:
    @property
    def pattern_id(self) -> str:
        return "plug_r1_fail_invoke"

    def execute(
        self,
        *,
        context: ToolInvocationPatternContext,
        invoker: ToolInvocationInvokerPort,
        planner: ToolInvocationPlannerPort,
        plan: ToolCallPlan | None,
        allowed_tool_ids: Sequence[str] | None,
        max_iterations: int,
        planner_input: str | list[object],
    ) -> ToolInvocationPatternResult:
        _ = planner, plan, allowed_tool_ids, max_iterations, planner_input
        request = ToolExecutionRequest(
            run_id=context.run_id,
            step_id="public-fail",
            tool_id="plug.fail",
            input=_FailIn(message="controlled failure"),
        )
        result = invoker.invoke_tool(agent_id=context.agent_id or "agent", request=request)
        assert not result.success
        return ToolInvocationPatternResult(
            pattern_id=self.pattern_id,
            stop_reason="legacy_single_pass",
            loop_iterations=1,
        )


class _ParallelInvokePublicPattern:
    @property
    def pattern_id(self) -> str:
        return "plug_r1_parallel_invoke"

    def execute(
        self,
        *,
        context: ToolInvocationPatternContext,
        invoker: ToolInvocationInvokerPort,
        planner: ToolInvocationPlannerPort,
        plan: ToolCallPlan | None,
        allowed_tool_ids: Sequence[str] | None,
        max_iterations: int,
        planner_input: str | list[object],
    ) -> ToolInvocationPatternResult:
        _ = planner, plan, allowed_tool_ids, max_iterations, planner_input
        agent_id = context.agent_id or "agent"

        def _call(step_id: str, value: int) -> None:
            request = ToolExecutionRequest(
                run_id=context.run_id,
                step_id=step_id,
                tool_id="plug.demo",
                input=_In(value=value),
            )
            invoker.invoke_tool(agent_id=agent_id, request=request)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(copy_context().run, _call, "parallel-a", 10),
                pool.submit(copy_context().run, _call, "parallel-b", 20),
            ]
            for future in futures:
                future.result()
        return ToolInvocationPatternResult(
            pattern_id=self.pattern_id,
            stop_reason="legacy_single_pass",
            loop_iterations=1,
        )


def _apply_plan_context_tool_traces(state, loop_result) -> None:
    """Mirror plan_context_invocation post-loop state merge."""
    if not loop_result.tool_traces:
        if state.context.config.tools_mode == "required":
            raise ToolsRequiredError(run_id=state.run_id)
    else:
        state.used_tools = True
        state.tool_traces = list(loop_result.tool_traces)


def test_public_pattern_single_invoke_via_bounded_tool_loop() -> None:
    invoker = _make_invoker()
    run_seed = "plug-r1-single"
    state = build_runtime_state_for_tests(run_id=run_seed)
    bridge = PublicToolInvocationPatternBridge(_SingleInvokePublicPattern())

    with canonical_governed_execution_scope(run_seed):
        loop_result = run_bounded_tool_loop(
            state=state,
            invoker=invoker,
            tool_planner=_NoopPlanner(),
            planner_input="invoke",
            allowed_tool_ids=("plug.demo",),
            max_iterations=1,
            pattern=bridge,
        )
        _apply_plan_context_tool_traces(state, loop_result)

    assert len(loop_result.tool_traces) == 1
    assert loop_result.tool_traces[0].tool_name == "plug.demo"
    assert loop_result.tool_traces[0].success is True
    assert state.used_tools is True
    assert len(state.tool_traces) == 1
    assert loop_result.aggregate is not None


def test_public_pattern_failed_invoke_still_records_trace() -> None:
    invoker = _make_invoker()
    run_seed = "plug-r1-fail"
    state = build_runtime_state_for_tests(run_id=run_seed)
    bridge = PublicToolInvocationPatternBridge(_FailingInvokePublicPattern())

    with canonical_governed_execution_scope(run_seed):
        loop_result = run_bounded_tool_loop(
            state=state,
            invoker=invoker,
            tool_planner=_NoopPlanner(),
            planner_input="invoke",
            allowed_tool_ids=("plug.fail",),
            max_iterations=1,
            pattern=bridge,
        )
        _apply_plan_context_tool_traces(state, loop_result)

    assert len(loop_result.tool_traces) == 1
    assert loop_result.tool_traces[0].success is False
    assert state.used_tools is True


def test_public_pattern_multiple_invokes_stable_order() -> None:
    invoker = _make_invoker()
    run_seed = "plug-r1-dual"
    state = build_runtime_state_for_tests(run_id=run_seed)
    bridge = PublicToolInvocationPatternBridge(_DualInvokePublicPattern())

    with canonical_governed_execution_scope(run_seed):
        loop_result = run_bounded_tool_loop(
            state=state,
            invoker=invoker,
            tool_planner=_NoopPlanner(),
            planner_input="invoke",
            allowed_tool_ids=("plug.demo",),
            max_iterations=1,
            pattern=bridge,
        )

    assert len(loop_result.tool_traces) == 2
    assert loop_result.tool_traces[0].arguments["value"] == 1
    assert loop_result.tool_traces[1].arguments["value"] == 2


def test_public_pattern_parallel_invokes_thread_safe_evidence() -> None:
    invoker = _make_invoker()
    run_seed = "plug-r1-parallel"
    state = build_runtime_state_for_tests(run_id=run_seed)
    bridge = PublicToolInvocationPatternBridge(_ParallelInvokePublicPattern())

    with canonical_governed_execution_scope(run_seed):
        loop_result = run_bounded_tool_loop(
            state=state,
            invoker=invoker,
            tool_planner=_NoopPlanner(),
            planner_input="invoke",
            allowed_tool_ids=("plug.demo",),
            max_iterations=1,
            pattern=bridge,
        )

    assert len(loop_result.tool_traces) == 2
    values = sorted(trace.arguments["value"] for trace in loop_result.tool_traces)
    assert values == [10, 20]


def test_tools_mode_required_passes_after_real_public_invoke() -> None:
    invoker = _make_invoker()
    run_seed = "plug-r1-required-pass"
    state = build_runtime_state_for_tests(run_id=run_seed)
    state.context.config.tools_mode = "required"
    bridge = PublicToolInvocationPatternBridge(_SingleInvokePublicPattern())

    with canonical_governed_execution_scope(run_seed):
        loop_result = run_bounded_tool_loop(
            state=state,
            invoker=invoker,
            tool_planner=_NoopPlanner(),
            planner_input="invoke",
            allowed_tool_ids=("plug.demo",),
            max_iterations=1,
            pattern=bridge,
        )
        _apply_plan_context_tool_traces(state, loop_result)

    assert state.used_tools is True


def test_tools_mode_required_raises_when_public_pattern_invokes_nothing() -> None:
    invoker = _make_invoker()
    run_seed = "plug-r1-required-empty"
    state = build_runtime_state_for_tests(run_id=run_seed)
    state.context.config.tools_mode = "required"
    bridge = PublicToolInvocationPatternBridge(_EmptyPublicPattern())

    with canonical_governed_execution_scope(run_seed):
        loop_result = run_bounded_tool_loop(
            state=state,
            invoker=invoker,
            tool_planner=_NoopPlanner(),
            planner_input="invoke",
            allowed_tool_ids=("plug.demo",),
            max_iterations=1,
            pattern=bridge,
        )
        with pytest.raises(ToolsRequiredError):
            _apply_plan_context_tool_traces(state, loop_result)


def test_side_effect_tool_runs_once_with_single_trace() -> None:
    _SIDE_EFFECT_COUNTER["count"] = 0
    invoker = _make_invoker()
    run_seed = "plug-r1-side-effect"
    state = build_runtime_state_for_tests(run_id=run_seed)

    class _SideEffectPattern:
        @property
        def pattern_id(self) -> str:
            return "plug_r1_side_effect"

        def execute(
            self,
            *,
            context: ToolInvocationPatternContext,
            invoker: ToolInvocationInvokerPort,
            planner: ToolInvocationPlannerPort,
            plan: ToolCallPlan | None,
            allowed_tool_ids: Sequence[str] | None,
            max_iterations: int,
            planner_input: str | list[object],
        ) -> ToolInvocationPatternResult:
            _ = planner, plan, allowed_tool_ids, max_iterations, planner_input
            request = ToolExecutionRequest(
                run_id=context.run_id,
                step_id="side-effect",
                tool_id="plug.side_effect",
                input=_SideEffectIn(token="once"),
            )
            invoker.invoke_tool(agent_id=context.agent_id or "agent", request=request)
            return ToolInvocationPatternResult(
                pattern_id=self.pattern_id,
                stop_reason="legacy_single_pass",
                loop_iterations=1,
            )

    bridge = PublicToolInvocationPatternBridge(_SideEffectPattern())

    with canonical_governed_execution_scope(run_seed):
        loop_result = run_bounded_tool_loop(
            state=state,
            invoker=invoker,
            tool_planner=_NoopPlanner(),
            planner_input="invoke",
            allowed_tool_ids=("plug.side_effect",),
            max_iterations=1,
            pattern=bridge,
        )

    assert _SIDE_EFFECT_COUNTER["count"] == 1
    assert len(loop_result.tool_traces) == 1
