# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-6 — bounded multi-iteration tool loop integration."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import json
import pytest
from pydantic import BaseModel

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.runtime.nexus.budget.budget_diagnostics import BudgetExceededDiagV1
from intergrax.runtime.nexus.budget.budget_enforcer import BudgetExceededError
from intergrax.runtime.nexus.budget.budget_models import BudgetEnforcementMode, BudgetPolicy, RunBudget
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.config_types import ToolsContextScope
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.runtime.nexus.tools.tool_loop import execute_planned_tool_calls, run_bounded_tool_loop
from intergrax.runtime.nexus.tools.native_planner_action_context import (
    PLANNER_ACTION_CONTEXT_TOOL_ID,
    NativePlannerActionContextError,
    NativePlannerRound,
    resolve_native_planner_protocol,
)
from intergrax.runtime.nexus.tools.investigation_proof import (
    InvestigationProofValidationError,
    collect_available_evidence_ids,
    investigation_native_planner_protocol_config,
)
from intergrax.runtime.nexus.tools.native_tool_plan_alignment import NativeToolPlanAlignmentError
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.nexus.tools.tool_planning_prompts import investigation_policy_prompt
from intergrax.runtime.nexus.tools.tool_planning_service import ToolPlanningService
from intergrax.runtime.nexus.tools.tool_planner_protocol import IterativeToolPlannerProtocol
from intergrax.tools.core.tool_plan import PlannedToolCall, ToolCallPlan
from intergrax.tools.core.tool_plan_decision import ToolPlanDecision
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.tool_executor import ToolHandler
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager, canonical_execution_identity_scope, tools_agent_make_contract

pytestmark = [pytest.mark.integration, pytest.mark.gate]

_INTEGRATION_RUN_ID = RunId("run_00000000000000000000000000000001")


def _invoke_bounded_tool_loop(**kwargs):
    from intergrax.contracts.execution_identity import require_active_execution_id
    from intergrax.runtime.execution.active_execution_budget import (
        bind_root_execution_budget,
        peek_active_execution_budget,
        reset_active_execution_budget,
    )
    from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger

    state = kwargs["state"]
    with canonical_execution_identity_scope(state.run_id):
        budget_token = None
        if peek_active_execution_budget() is None:
            budget_token = bind_root_execution_budget(
                execution_id=require_active_execution_id(),
                ledger=create_execution_budget_ledger(None),
            )
        try:
            return run_bounded_tool_loop(**kwargs)
        finally:
            if budget_token is not None:
                reset_active_execution_budget(budget_token)


def _invoke_planned_tool_calls(**kwargs):
    from intergrax.contracts.execution_identity import require_active_execution_id
    from intergrax.runtime.execution.active_execution_budget import (
        bind_root_execution_budget,
        peek_active_execution_budget,
        reset_active_execution_budget,
    )
    from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger

    state = kwargs["state"]
    with canonical_execution_identity_scope(state.run_id):
        budget_token = None
        if peek_active_execution_budget() is None:
            budget_token = bind_root_execution_budget(
                execution_id=require_active_execution_id(),
                ledger=create_execution_budget_ledger(None),
            )
        try:
            return execute_planned_tool_calls(**kwargs)
        finally:
            if budget_token is not None:
                reset_active_execution_budget(budget_token)


_BEYOND_PREVIEW_MARKER = "ENG1_TAIL_MARKER"
_PREVIEW_BOUND = 400


def _decision_note(*basis_ids: str, purpose: str) -> str:
    basis = ",".join(basis_ids)
    return f"EVIDENCE_BASIS: {basis}\nPURPOSE: {purpose}"


def _action_context_call(
    *basis_ids: str,
    purpose: str,
    call_id: str = "ann-ctx",
) -> LLMToolCall:
    return LLMToolCall(
        id=call_id,
        name=PLANNER_ACTION_CONTEXT_TOOL_ID,
        arguments_json=json.dumps(
            {
                "evidence_basis_references": list(basis_ids),
                "purpose": purpose,
            }
        ),
    )


def _prior_evidence_references(messages: list[ChatMessage]) -> tuple[str, ...]:
    return collect_available_evidence_ids(messages)


def _planner_round_from_response(
    response: LLMAdapterResponse,
    tool_plan: ToolCallPlan,
    *,
    messages: list[ChatMessage],
) -> NativePlannerRound:
    protocol_config = investigation_native_planner_protocol_config(messages)
    action_context, business_calls = resolve_native_planner_protocol(
        response.tool_calls,
        protocol_config=protocol_config,
    )
    return NativePlannerRound(
        response=response,
        business_tool_calls=business_calls,
        tool_plan=tool_plan,
        action_context=action_context,
    )


class _InA(BaseModel):
    value: int = 1


class _OutA(BaseModel):
    result: int = 0


class _HandlerA(ToolHandler[_InA, _OutA]):
    def execute(self, request: ToolExecutionRequest[_InA]) -> _OutA:
        return _OutA(result=request.input.value)


class _LongOut(BaseModel):
    padding: str
    decision_token: str


class _LongIn(BaseModel):
    pass


class _LongHandler(ToolHandler[_LongIn, _LongOut]):
    def execute(self, request: ToolExecutionRequest[_LongIn]) -> _LongOut:
        _ = request
        return _LongOut(padding="x" * 500, decision_token=_BEYOND_PREVIEW_MARKER)


class _FailIn(BaseModel):
    pass


class _FailOut(BaseModel):
    ok: bool = True


class _FailHandler(ToolHandler[_FailIn, _FailOut]):
    def execute(self, request: ToolExecutionRequest[_FailIn]) -> _FailOut:
        _ = request
        raise RuntimeError("x" * 400 + _BEYOND_PREVIEW_MARKER)


class _TwoRoundLLM(FakeLLMAdapter):
    def __init__(self) -> None:
        super().__init__(fixed_text="")
        self._round = 0

    def supports_tools(self) -> bool:
        return True

    def generate_with_tools(self, messages, tools_schema, **kwargs):  # type: ignore[no-untyped-def]
        _ = messages, tools_schema, kwargs
        self._round += 1
        if self._round == 1:
            return LLMAdapterResponse(
                content="",
                tool_calls=(
                    LLMToolCall.from_openai_shape(
                        call_id="tc-1",
                        name="alpha.tool",
                        arguments={"value": 7},
                    ),
                ),
            )
        return LLMAdapterResponse(content="done", tool_calls=())


def _runtime_state(llm: FakeLLMAdapter) -> RuntimeState:
    config = RuntimeConfig(
        llm_adapter=llm,
        production_mode=False,
        enable_rag=False,
        enable_websearch=False,
        tools_context_scope=ToolsContextScope.CURRENT_MESSAGE_ONLY,
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
            task_id=TaskId("task_00000000000000000000000000000001"),
            run_id=_INTEGRATION_RUN_ID,
            message="use tool",
        ),
        run_id=_INTEGRATION_RUN_ID,
        messages_for_llm=[ChatMessage(role="user", content="use tool")],
    )


def test_bounded_tool_loop_two_iterations_native_messages() -> None:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("alpha.tool", _InA, _OutA),
        _HandlerA(),
    )
    llm = _TwoRoundLLM()
    state = _runtime_state(llm)
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = ToolPlanningService(llm=llm, tools=registry)

    result = _invoke_bounded_tool_loop(
        state=state,
        invoker=invoker,
        tool_planner=planner,
        planner_input=[ChatMessage(role="user", content="use tool")],
        allowed_tool_ids=("alpha.tool",),
        max_iterations=2,
    )

    assert result.loop_iterations == 2
    assert len(result.tool_traces) == 1
    assert result.tool_traces[0].tool_name == "alpha.tool"
    assert result.used_native_tool_messages is True
    assert any(msg.role == "tool" for msg in result.appended_messages)
    assert llm._round == 2
    assert result.stop_reason == "planner_final_answer"


class _EmptyPlanLLM(FakeLLMAdapter):
    def supports_tools(self) -> bool:
        return True

    def generate_with_tools(self, messages, tools_schema, **kwargs):  # type: ignore[no-untyped-def]
        _ = messages, tools_schema, kwargs
        return LLMAdapterResponse(content="", tool_calls=())


class _AlwaysToolLLM(FakeLLMAdapter):
    def __init__(self) -> None:
        super().__init__(fixed_text="")
        self._round = 0

    def supports_tools(self) -> bool:
        return True

    def generate_with_tools(self, messages, tools_schema, **kwargs):  # type: ignore[no-untyped-def]
        _ = tools_schema, kwargs
        self._round += 1
        if self._round == 1:
            return LLMAdapterResponse(
                content="",
                tool_calls=(
                    LLMToolCall.from_openai_shape(
                        call_id=f"tc-{self._round}",
                        name="alpha.tool",
                        arguments={"value": self._round},
                    ),
                ),
            )
        prior_basis = _prior_evidence_references(list(messages))
        return LLMAdapterResponse(
            content="",
            tool_calls=(
                _action_context_call(*prior_basis, purpose="continue investigation"),
                LLMToolCall.from_openai_shape(
                    call_id=f"tc-{self._round}",
                    name="alpha.tool",
                    arguments={"value": self._round},
                ),
            ),
        )


class _PlannerExplodedError(RuntimeError):
    pass


class _FailAfterOneRoundLLM(FakeLLMAdapter):
    def __init__(self) -> None:
        super().__init__(fixed_text="")
        self._round = 0

    def supports_tools(self) -> bool:
        return True

    def generate_with_tools(self, messages, tools_schema, **kwargs):  # type: ignore[no-untyped-def]
        _ = messages, tools_schema, kwargs
        self._round += 1
        if self._round == 1:
            return LLMAdapterResponse(
                content="",
                tool_calls=(
                    LLMToolCall.from_openai_shape(
                        call_id="tc-1",
                        name="alpha.tool",
                        arguments={"value": 1},
                    ),
                ),
            )
        raise _PlannerExplodedError("planner exploded deterministically")


def test_bounded_tool_loop_empty_tool_calls_stop_reason() -> None:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("alpha.tool", _InA, _OutA),
        _HandlerA(),
    )
    llm = _EmptyPlanLLM()
    state = _runtime_state(llm)
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = ToolPlanningService(llm=llm, tools=registry)

    result = _invoke_bounded_tool_loop(
        state=state,
        invoker=invoker,
        tool_planner=planner,
        planner_input=[ChatMessage(role="user", content="use tool")],
        allowed_tool_ids=("alpha.tool",),
        max_iterations=3,
    )

    assert result.loop_iterations == 1
    assert result.tool_traces == []
    assert result.stop_reason == "empty_tool_calls"


def test_bounded_tool_loop_max_iterations_stop_reason() -> None:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("alpha.tool", _InA, _OutA),
        _HandlerA(),
    )
    llm = _AlwaysToolLLM()
    state = _runtime_state(llm)
    state.context.config.max_tool_iterations = 3
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = ToolPlanningService(llm=llm, tools=registry)

    result = _invoke_bounded_tool_loop(
        state=state,
        invoker=invoker,
        tool_planner=planner,
        planner_input=[ChatMessage(role="user", content="use tool")],
        allowed_tool_ids=("alpha.tool",),
        max_iterations=3,
    )

    assert result.loop_iterations == 3
    assert len(result.tool_traces) == 3
    assert result.stop_reason == "max_iterations"
    assert llm._round == 3


def test_bounded_tool_loop_planner_failure_propagates() -> None:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("alpha.tool", _InA, _OutA),
        _HandlerA(),
    )
    llm = _FailAfterOneRoundLLM()
    state = _runtime_state(llm)
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = ToolPlanningService(llm=llm, tools=registry)

    with pytest.raises(_PlannerExplodedError, match="planner exploded deterministically"):
        _invoke_bounded_tool_loop(
            state=state,
            invoker=invoker,
            tool_planner=planner,
            planner_input=[ChatMessage(role="user", content="use tool")],
            allowed_tool_ids=("alpha.tool",),
            max_iterations=3,
        )


def test_runbudget_max_tool_calls_aborts_after_second_invocation() -> None:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("alpha.tool", _InA, _OutA),
        _CountingHandler(),
    )
    _CountingHandler.invocations = 0
    llm = _AlwaysToolLLM()
    state = _runtime_state(llm)
    state.context.config.run_budget = RunBudget(max_tool_calls=1)
    state.context.config.budget_policy = BudgetPolicy(enforcement_mode=BudgetEnforcementMode.ABORT)
    state.context.config.max_tool_iterations = 5
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = ToolPlanningService(llm=llm, tools=registry)

    with pytest.raises(BudgetExceededError, match="max_tool_calls"):
        _invoke_bounded_tool_loop(
            state=state,
            invoker=invoker,
            tool_planner=planner,
            planner_input=[ChatMessage(role="user", content="use tool")],
            allowed_tool_ids=("alpha.tool",),
            max_iterations=5,
        )

    assert _CountingHandler.invocations == 1
    assert llm._round == 2
    assert len(state.tool_traces) == 1
    budget_events = [
        event
        for event in state.trace_events
        if event.payload is not None
        and isinstance(event.payload, BudgetExceededDiagV1)
        and event.payload.budget_name == "max_tool_calls"
    ]
    assert len(budget_events) == 1
    assert budget_events[0].payload.limit == 1
    assert budget_events[0].payload.actual == 2


def test_runbudget_max_tool_calls_allows_full_loop_without_double_count() -> None:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("alpha.tool", _InA, _OutA),
        _CountingHandler(),
    )
    _CountingHandler.invocations = 0
    llm = _TwoRoundLLM()
    state = _runtime_state(llm)
    state.context.config.run_budget = RunBudget(max_tool_calls=10)
    state.context.config.budget_policy = BudgetPolicy(enforcement_mode=BudgetEnforcementMode.ABORT)
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = ToolPlanningService(llm=llm, tools=registry)

    result = _invoke_bounded_tool_loop(
        state=state,
        invoker=invoker,
        tool_planner=planner,
        planner_input=[ChatMessage(role="user", content="use tool")],
        allowed_tool_ids=("alpha.tool",),
        max_iterations=2,
    )

    assert _CountingHandler.invocations == 1
    assert len(result.tool_traces) == 1
    assert len(state.tool_traces) == 1
    assert result.tool_traces[0] is state.tool_traces[0]


class _SlowReadOnlyHandler(ToolHandler[_InA, _OutA]):
    def execute(self, request: ToolExecutionRequest[_InA]) -> _OutA:
        time.sleep(0.05)
        return _OutA(result=request.input.value)


def test_parallel_read_only_max_tool_calls_enforced_under_invoke_lock() -> None:
    registry = ToolRegistry()
    for tool_id in ("read.a", "read.b"):
        registry.register(
            tools_agent_make_contract(tool_id, _InA, _OutA),
            _SlowReadOnlyHandler(),
        )
    state = _runtime_state(FakeLLMAdapter())
    state.context.config.run_budget = RunBudget(max_tool_calls=1)
    state.context.config.budget_policy = BudgetPolicy(enforcement_mode=BudgetEnforcementMode.ABORT)
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    calls = [
        PlannedToolCall(step_id="s1", tool_id="read.a", input=_InA(value=1)),
        PlannedToolCall(step_id="s2", tool_id="read.b", input=_InA(value=2)),
    ]

    with pytest.raises(BudgetExceededError, match="max_tool_calls"):
        _invoke_planned_tool_calls(
            state=state,
            invoker=invoker,
            calls=calls,
            idempotency_prefix="budget-parallel",
            max_parallel_read_only=2,
        )

    assert len(state.tool_traces) == 2
    budget_events = [
        event
        for event in state.trace_events
        if event.payload is not None
        and isinstance(event.payload, BudgetExceededDiagV1)
        and event.payload.budget_name == "max_tool_calls"
    ]
    assert len(budget_events) == 1
    assert budget_events[0].payload.limit == 1
    assert budget_events[0].payload.actual == 2


class _LongOutputTwoRoundLLM(FakeLLMAdapter):
    def __init__(self) -> None:
        super().__init__(fixed_text="")
        self._round = 0
        self.second_round_saw_marker = False

    def supports_tools(self) -> bool:
        return True

    def generate_with_tools(self, messages, tools_schema, **kwargs):  # type: ignore[no-untyped-def]
        _ = tools_schema, kwargs
        self._round += 1
        if self._round == 1:
            return LLMAdapterResponse(
                content="",
                tool_calls=(
                    LLMToolCall.from_openai_shape(
                        call_id="tc-long",
                        name="long.tool",
                        arguments={},
                    ),
                ),
            )
        tool_messages = [msg for msg in messages if msg.role == "tool"]
        assert len(tool_messages) == 1
        tool_body = tool_messages[0].content or ""
        self.second_round_saw_marker = _BEYOND_PREVIEW_MARKER in tool_body
        if not self.second_round_saw_marker:
            return LLMAdapterResponse(content="missing tail marker", tool_calls=())
        return LLMAdapterResponse(content=f"confirmed:{_BEYOND_PREVIEW_MARKER}", tool_calls=())


class _FailThenRecoverLLM(FakeLLMAdapter):
    def __init__(self) -> None:
        super().__init__(fixed_text="")
        self._round = 0
        self.second_round_saw_marker = False

    def supports_tools(self) -> bool:
        return True

    def generate_with_tools(self, messages, tools_schema, **kwargs):  # type: ignore[no-untyped-def]
        _ = tools_schema, kwargs
        self._round += 1
        if self._round == 1:
            return LLMAdapterResponse(
                content="",
                tool_calls=(
                    LLMToolCall.from_openai_shape(
                        call_id="tc-fail",
                        name="fail.tool",
                        arguments={},
                    ),
                ),
            )
        tool_messages = [msg for msg in messages if msg.role == "tool"]
        assert len(tool_messages) == 1
        tool_body = tool_messages[0].content or ""
        self.second_round_saw_marker = _BEYOND_PREVIEW_MARKER in tool_body
        if not self.second_round_saw_marker:
            return LLMAdapterResponse(content="missing tail marker", tool_calls=())
        return LLMAdapterResponse(content=f"confirmed:{_BEYOND_PREVIEW_MARKER}", tool_calls=())


def test_model_facing_tool_result_preserves_output_beyond_trace_preview() -> None:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("long.tool", _LongIn, _LongOut),
        _LongHandler(),
    )
    llm = _LongOutputTwoRoundLLM()
    state = _runtime_state(llm)
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = ToolPlanningService(llm=llm, tools=registry)

    result = _invoke_bounded_tool_loop(
        state=state,
        invoker=invoker,
        tool_planner=planner,
        planner_input=[ChatMessage(role="user", content="use long tool")],
        allowed_tool_ids=("long.tool",),
        max_iterations=2,
    )

    trace = result.tool_traces[0]
    full_json = _LongOut(padding="x" * 500, decision_token=_BEYOND_PREVIEW_MARKER).model_dump_json()
    tool_messages = [msg for msg in result.appended_messages if msg.role == "tool"]

    assert trace.output_preview is not None
    assert len(trace.output_preview) <= _PREVIEW_BOUND
    assert _BEYOND_PREVIEW_MARKER not in trace.output_preview
    assert len(tool_messages) == 1
    tool_content = tool_messages[0].content or ""
    assert "EVIDENCE_REF: observation.long.tool." in tool_content
    assert f"{_INTEGRATION_RUN_ID}:loop1:tool" in tool_content
    tool_payload = json.loads(tool_content.split("\n", 1)[1])
    full_payload = json.loads(full_json)
    assert tool_payload["decision_token"] == full_payload["decision_token"]
    assert tool_payload["padding"] == full_payload["padding"]
    assert "evidence_reference" not in tool_payload
    assert llm.second_round_saw_marker is True
    assert result.stop_reason == "planner_final_answer"


def test_failed_tool_keeps_bounded_trace_and_model_facing_error() -> None:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("fail.tool", _FailIn, _FailOut),
        _FailHandler(),
    )
    llm = _FailThenRecoverLLM()
    state = _runtime_state(llm)
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = ToolPlanningService(llm=llm, tools=registry)

    result = _invoke_bounded_tool_loop(
        state=state,
        invoker=invoker,
        tool_planner=planner,
        planner_input=[ChatMessage(role="user", content="use failing tool")],
        allowed_tool_ids=("fail.tool",),
        max_iterations=2,
    )

    trace = result.tool_traces[0]
    tool_messages = [msg for msg in result.appended_messages if msg.role == "tool"]
    full_error = f"RuntimeError: {'x' * 400 + _BEYOND_PREVIEW_MARKER}"

    assert trace.success is False
    assert trace.output_preview is None
    assert trace.error_message is not None
    assert len(trace.error_message) <= _PREVIEW_BOUND
    assert _BEYOND_PREVIEW_MARKER not in trace.error_message
    assert len(tool_messages) == 1
    assert tool_messages[0].content == full_error
    assert _BEYOND_PREVIEW_MARKER in tool_messages[0].content
    assert llm.second_round_saw_marker is True
    assert result.stop_reason == "planner_final_answer"


class _BaseOnlyPlanner:
    """Implements ToolPlannerProtocol only — no native iterative rounds."""

    def plan_tools(
        self,
        input_data,
        context=None,
        *,
        run_id: str,
        allowed_tool_ids=None,
        **kwargs,
    ) -> ToolPlanDecision:
        _ = input_data, context, run_id, allowed_tool_ids, kwargs
        return ToolPlanDecision(
            final_answer=None,
            tool_plan=ToolCallPlan(
                calls=[
                    PlannedToolCall(
                        step_id="tool",
                        tool_id="alpha.tool",
                        input=_InA(value=3),
                    )
                ]
            ),
            messages=[],
        )


class _CustomIterativePlanner:
    """Custom iterative planner — not ToolPlanningService."""

    def __init__(self) -> None:
        self._round = 0

    def plan_tools(
        self,
        input_data,
        context=None,
        *,
        run_id: str,
        allowed_tool_ids=None,
        **kwargs,
    ) -> ToolPlanDecision:
        _ = input_data, context, run_id, allowed_tool_ids, kwargs
        return ToolPlanDecision(
            final_answer=None,
            tool_plan=ToolCallPlan(calls=[]),
            messages=[],
        )

    def plan_native_round(
        self,
        messages,
        *,
        allowed_tool_ids=None,
        run_id=None,
        tool_choice=None,
        **kwargs,
    ):
        _ = messages, allowed_tool_ids, run_id, tool_choice, kwargs
        self._round += 1
        if self._round == 1:
            response = LLMAdapterResponse(
                content="",
                tool_calls=(
                    LLMToolCall.from_openai_shape(
                        call_id="custom-tc-1",
                        name="alpha.tool",
                        arguments={"value": 11},
                    ),
                ),
            )
            tool_plan = ToolCallPlan(
                calls=[
                    PlannedToolCall(
                        step_id="tool",
                        tool_id="alpha.tool",
                        input=_InA(value=11),
                    )
                ]
            )
            return _planner_round_from_response(
                response,
                tool_plan,
                messages=list(messages),
            )
        return NativePlannerRound(
            response=LLMAdapterResponse(content="custom done", tool_calls=()),
            business_tool_calls=(),
            tool_plan=ToolCallPlan(calls=[]),
            action_context=None,
        )


def test_custom_iterative_planner_runs_bounded_loop_without_degrading() -> None:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("alpha.tool", _InA, _OutA),
        _HandlerA(),
    )
    llm = _TwoRoundLLM()
    state = _runtime_state(llm)
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = _CustomIterativePlanner()
    assert not isinstance(planner, ToolPlanningService)
    assert isinstance(planner, IterativeToolPlannerProtocol)

    result = _invoke_bounded_tool_loop(
        state=state,
        invoker=invoker,
        tool_planner=planner,
        planner_input=[ChatMessage(role="user", content="use tool")],
        allowed_tool_ids=("alpha.tool",),
        max_iterations=2,
    )

    assert result.loop_iterations == 2
    assert len(result.tool_traces) == 1
    assert result.tool_traces[0].tool_name == "alpha.tool"
    assert result.tool_traces[0].output_preview is not None
    assert result.stop_reason == "planner_final_answer"
    assert result.used_native_tool_messages is True
    assert planner._round == 2


def test_base_only_planner_rejected_for_multi_iteration_bounded_loop() -> None:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("alpha.tool", _InA, _OutA),
        _HandlerA(),
    )
    llm = _TwoRoundLLM()
    state = _runtime_state(llm)
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = _BaseOnlyPlanner()
    assert not isinstance(planner, IterativeToolPlannerProtocol)

    with pytest.raises(
        TypeError,
        match="Bounded iterative tool invocation \\(max_iterations > 1\\) requires",
    ):
        _invoke_bounded_tool_loop(
            state=state,
            invoker=invoker,
            tool_planner=planner,
            planner_input=[ChatMessage(role="user", content="use tool")],
            allowed_tool_ids=("alpha.tool",),
            max_iterations=2,
        )


def test_base_only_planner_single_iteration_still_uses_single_pass() -> None:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("alpha.tool", _InA, _OutA),
        _HandlerA(),
    )
    llm = _TwoRoundLLM()
    state = _runtime_state(llm)
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = _BaseOnlyPlanner()

    result = _invoke_bounded_tool_loop(
        state=state,
        invoker=invoker,
        tool_planner=planner,
        planner_input=[ChatMessage(role="user", content="use tool")],
        allowed_tool_ids=("alpha.tool",),
        max_iterations=1,
    )

    assert result.loop_iterations == 1
    assert len(result.tool_traces) == 1
    assert result.tool_traces[0].tool_name == "alpha.tool"
    assert result.stop_reason == "legacy_single_pass"
    assert result.used_native_tool_messages is False


class _CountingHandler(ToolHandler[_InA, _OutA]):
    invocations: int = 0

    def execute(self, request: ToolExecutionRequest[_InA]) -> _OutA:
        _CountingHandler.invocations += 1
        return _OutA(result=request.input.value)


class _MultiCallRoundPlanner:
    def __init__(self, *, rounds: int) -> None:
        self._round = 0
        self._max_rounds = rounds

    def plan_tools(
        self,
        input_data,
        context=None,
        *,
        run_id: str,
        allowed_tool_ids=None,
        **kwargs,
    ) -> ToolPlanDecision:
        _ = input_data, context, run_id, allowed_tool_ids, kwargs
        return ToolPlanDecision(
            final_answer=None,
            tool_plan=ToolCallPlan(calls=[]),
            messages=[],
        )

    def plan_native_round(
        self,
        messages,
        *,
        allowed_tool_ids=None,
        run_id=None,
        tool_choice=None,
        **kwargs,
    ):
        _ = messages, allowed_tool_ids, run_id, tool_choice, kwargs
        self._round += 1
        if self._round <= self._max_rounds:
            calls = [
                PlannedToolCall(
                    step_id=f"tool-{index}",
                    tool_id="alpha.tool",
                    input=_InA(value=index),
                )
                for index in range(self._round + 1)
            ]
            tool_calls = tuple(
                LLMToolCall.from_openai_shape(
                    call_id=f"tc-{index}",
                    name="alpha.tool",
                    arguments={"value": index},
                )
                for index in range(self._round + 1)
            )
            if self._round == 1:
                provider_calls = tool_calls
            else:
                prior_basis = _prior_evidence_references(list(messages))
                provider_calls = (
                    _action_context_call(*prior_basis, purpose="continue per-round batch"),
                    *tool_calls,
                )
            response = LLMAdapterResponse(content="", tool_calls=provider_calls)
            tool_plan = ToolCallPlan(calls=calls)
            return _planner_round_from_response(
                response,
                tool_plan,
                messages=list(messages),
            )
        return NativePlannerRound(
            response=LLMAdapterResponse(content="done", tool_calls=()),
            business_tool_calls=(),
            tool_plan=ToolCallPlan(calls=[]),
            action_context=None,
        )


def test_per_round_tool_call_limit_rejects_before_invocation() -> None:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("alpha.tool", _InA, _OutA),
        _CountingHandler(),
    )
    _CountingHandler.invocations = 0
    llm = _TwoRoundLLM()
    state = _runtime_state(llm)
    state.context.config.max_tool_calls_per_round = 1
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = _MultiCallRoundPlanner(rounds=3)

    with pytest.raises(ValueError, match="max_tool_calls_per_round"):
        _invoke_bounded_tool_loop(
            state=state,
            invoker=invoker,
            tool_planner=planner,
            planner_input=[ChatMessage(role="user", content="use tool")],
            allowed_tool_ids=("alpha.tool",),
            max_iterations=3,
        )

    assert _CountingHandler.invocations == 0


def test_runbudget_planner_iterations_emits_diag_and_aborts() -> None:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("alpha.tool", _InA, _OutA),
        _HandlerA(),
    )
    llm = _AlwaysToolLLM()
    state = _runtime_state(llm)
    state.context.config.run_budget = RunBudget(max_planner_iterations=1)
    state.context.config.budget_policy = BudgetPolicy(enforcement_mode=BudgetEnforcementMode.ABORT)
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = ToolPlanningService(llm=llm, tools=registry)

    with pytest.raises(BudgetExceededError, match="max_planner_iterations"):
        _invoke_bounded_tool_loop(
            state=state,
            invoker=invoker,
            tool_planner=planner,
            planner_input=[ChatMessage(role="user", content="use tool")],
            allowed_tool_ids=("alpha.tool",),
            max_iterations=3,
        )

    assert llm._round == 1
    budget_events = [
        event
        for event in state.trace_events
        if event.payload is not None
        and isinstance(event.payload, BudgetExceededDiagV1)
        and event.payload.budget_name == "max_planner_iterations"
    ]
    assert len(budget_events) == 1
    assert budget_events[0].payload.limit == 1
    assert budget_events[0].payload.actual == 2


def test_wall_time_budget_aborts_before_next_planner_round() -> None:
    from datetime import datetime, timedelta, timezone

    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("alpha.tool", _InA, _OutA),
        _HandlerA(),
    )
    llm = _AlwaysToolLLM()
    state = _runtime_state(llm)
    state.context.config.run_budget = RunBudget(max_wall_time_seconds=1.0)
    state.context.config.budget_policy = BudgetPolicy(enforcement_mode=BudgetEnforcementMode.ABORT)
    state.started_at_utc = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = ToolPlanningService(llm=llm, tools=registry)

    with pytest.raises(BudgetExceededError, match="max_wall_time_seconds"):
        _invoke_bounded_tool_loop(
            state=state,
            invoker=invoker,
            tool_planner=planner,
            planner_input=[ChatMessage(role="user", content="use tool")],
            allowed_tool_ids=("alpha.tool",),
            max_iterations=3,
        )

    assert llm._round == 1


class _RepeatCallPlanner:
    def __init__(self, *, rounds: int, value: int) -> None:
        self._round = 0
        self._max_rounds = rounds
        self._value = value

    def plan_tools(
        self,
        input_data,
        context=None,
        *,
        run_id: str,
        allowed_tool_ids=None,
        **kwargs,
    ) -> ToolPlanDecision:
        _ = input_data, context, run_id, allowed_tool_ids, kwargs
        return ToolPlanDecision(
            final_answer=None,
            tool_plan=ToolCallPlan(calls=[]),
            messages=[],
        )

    def plan_native_round(
        self,
        messages,
        *,
        allowed_tool_ids=None,
        run_id=None,
        tool_choice=None,
        **kwargs,
    ):
        _ = messages, allowed_tool_ids, run_id, tool_choice, kwargs
        self._round += 1
        if self._round <= self._max_rounds:
            prior_basis = _prior_evidence_references(list(messages))
            business_call = LLMToolCall.from_openai_shape(
                call_id=f"tc-{self._round}",
                name="alpha.tool",
                arguments={"value": self._value},
            )
            if self._round == 1:
                provider_calls = (business_call,)
            else:
                provider_calls = (
                    _action_context_call(*prior_basis, purpose="repeat check"),
                    business_call,
                )
            response = LLMAdapterResponse(content="", tool_calls=provider_calls)
            tool_plan = ToolCallPlan(
                calls=[
                    PlannedToolCall(
                        step_id="tool",
                        tool_id="alpha.tool",
                        input=_InA(value=self._value),
                    )
                ]
            )
            return _planner_round_from_response(
                response,
                tool_plan,
                messages=list(messages),
            )
        return NativePlannerRound(
            response=LLMAdapterResponse(content="done", tool_calls=()),
            business_tool_calls=(),
            tool_plan=ToolCallPlan(calls=[]),
            action_context=None,
        )


def test_identical_call_repeat_limit_rejects_before_third_execution() -> None:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("alpha.tool", _InA, _OutA),
        _CountingHandler(),
    )
    _CountingHandler.invocations = 0
    state = _runtime_state(_TwoRoundLLM())
    state.context.config.max_identical_tool_call_repeats = 2
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = _RepeatCallPlanner(rounds=3, value=7)

    with pytest.raises(RuntimeError, match="max_identical_tool_call_repeats"):
        _invoke_bounded_tool_loop(
            state=state,
            invoker=invoker,
            tool_planner=planner,
            planner_input=[ChatMessage(role="user", content="repeat")],
            allowed_tool_ids=("alpha.tool",),
            max_iterations=3,
        )

    assert _CountingHandler.invocations == 2


class _AlternatingInputPlanner:
    def __init__(self) -> None:
        self._round = 0
        self.sequence = (7, 8, 7)

    def plan_tools(
        self,
        input_data,
        context=None,
        *,
        run_id: str,
        allowed_tool_ids=None,
        **kwargs,
    ) -> ToolPlanDecision:
        _ = input_data, context, run_id, allowed_tool_ids, kwargs
        return ToolPlanDecision(
            final_answer=None,
            tool_plan=ToolCallPlan(calls=[]),
            messages=[],
        )

    def plan_native_round(
        self,
        messages,
        *,
        allowed_tool_ids=None,
        run_id=None,
        tool_choice=None,
        **kwargs,
    ):
        _ = messages, allowed_tool_ids, run_id, tool_choice, kwargs
        self._round += 1
        if self._round <= len(self.sequence):
            value = self.sequence[self._round - 1]
            prior_basis = _prior_evidence_references(list(messages))
            business_call = LLMToolCall.from_openai_shape(
                call_id=f"tc-{self._round}",
                name="alpha.tool",
                arguments={"value": value},
            )
            if self._round == 1:
                provider_calls = (business_call,)
            else:
                provider_calls = (
                    _action_context_call(*prior_basis, purpose="alternate input check"),
                    business_call,
                )
            response = LLMAdapterResponse(content="", tool_calls=provider_calls)
            tool_plan = ToolCallPlan(
                calls=[
                    PlannedToolCall(
                        step_id="tool",
                        tool_id="alpha.tool",
                        input=_InA(value=value),
                    )
                ]
            )
            return _planner_round_from_response(
                response,
                tool_plan,
                messages=list(messages),
            )
        return NativePlannerRound(
            response=LLMAdapterResponse(content="done", tool_calls=()),
            business_tool_calls=(),
            tool_plan=ToolCallPlan(calls=[]),
            action_context=None,
        )


def test_identical_call_guard_tracks_inputs_independently() -> None:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("alpha.tool", _InA, _OutA),
        _HandlerA(),
    )
    state = _runtime_state(_TwoRoundLLM())
    state.context.config.max_identical_tool_call_repeats = 2
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = _AlternatingInputPlanner()

    result = _invoke_bounded_tool_loop(
        state=state,
        invoker=invoker,
        tool_planner=planner,
        planner_input=[ChatMessage(role="user", content="alternate")],
        allowed_tool_ids=("alpha.tool",),
        max_iterations=4,
    )

    assert result.stop_reason == "planner_final_answer"
    assert len(result.tool_traces) == 3
    assert [trace.arguments["value"] for trace in result.tool_traces] == [7, 8, 7]


class _PartialFailureHandler(ToolHandler[_InA, _OutA]):
    def execute(self, request: ToolExecutionRequest[_InA]) -> _OutA:
        if request.input.value == 0:
            raise RuntimeError("soft tool failure")
        return _OutA(result=request.input.value)


class _MixedOutcomeRoundPlanner:
    def __init__(self) -> None:
        self._round = 0

    def plan_tools(
        self,
        input_data,
        context=None,
        *,
        run_id: str,
        allowed_tool_ids=None,
        **kwargs,
    ) -> ToolPlanDecision:
        _ = input_data, context, run_id, allowed_tool_ids, kwargs
        return ToolPlanDecision(
            final_answer=None,
            tool_plan=ToolCallPlan(calls=[]),
            messages=[],
        )

    def plan_native_round(
        self,
        messages,
        *,
        allowed_tool_ids=None,
        run_id=None,
        tool_choice=None,
        **kwargs,
    ):
        _ = allowed_tool_ids, run_id, tool_choice, kwargs
        self._round += 1
        if self._round == 1:
            response = LLMAdapterResponse(
                content="",
                tool_calls=(
                    LLMToolCall.from_openai_shape(
                        call_id="tc-ok",
                        name="alpha.tool",
                        arguments={"value": 5},
                    ),
                    LLMToolCall.from_openai_shape(
                        call_id="tc-fail",
                        name="alpha.tool",
                        arguments={"value": 0},
                    ),
                ),
            )
            tool_plan = ToolCallPlan(
                calls=[
                    PlannedToolCall(
                        step_id="tool-ok",
                        tool_id="alpha.tool",
                        input=_InA(value=5),
                    ),
                    PlannedToolCall(
                        step_id="tool-fail",
                        tool_id="alpha.tool",
                        input=_InA(value=0),
                    ),
                ]
            )
            return _planner_round_from_response(
                response,
                tool_plan,
                messages=list(messages),
            )
        tool_messages = [msg for msg in messages if msg.role == "tool"]
        assert len(tool_messages) == 2
        assert any("soft tool failure" in (msg.content or "") for msg in tool_messages)
        assert any('"result":5' in (msg.content or "") for msg in tool_messages)
        return NativePlannerRound(
            response=LLMAdapterResponse(content="mixed recovered", tool_calls=()),
            business_tool_calls=(),
            tool_plan=ToolCallPlan(calls=[]),
            action_context=None,
        )


def test_partial_tool_failure_continues_with_both_observations() -> None:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("alpha.tool", _InA, _OutA),
        _PartialFailureHandler(),
    )
    state = _runtime_state(_TwoRoundLLM())
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = _MixedOutcomeRoundPlanner()

    result = _invoke_bounded_tool_loop(
        state=state,
        invoker=invoker,
        tool_planner=planner,
        planner_input=[ChatMessage(role="user", content="mixed")],
        allowed_tool_ids=("alpha.tool",),
        max_iterations=2,
    )

    assert result.stop_reason == "planner_final_answer"
    assert len(result.tool_traces) == 2
    assert result.tool_traces[0].success is True
    assert result.tool_traces[1].success is False
    tool_messages = [msg for msg in result.appended_messages if msg.role == "tool"]
    assert len(tool_messages) == 2


_INVESTIGATION_POLICY_MARKER = "Investigation and evidence policy"
_NON_NATIVE_JSON_PROTOCOL_MARKER = "You do not have native tool-calling"


class _ProbeInA(BaseModel):
    query: str = "status"


class _ProbeOutA(BaseModel):
    status: str = "EVIDENCE_A"


class _ProbeInB(BaseModel):
    confirm: bool = True


class _ProbeOutB(BaseModel):
    status: str = "EVIDENCE_B"


class _ProbeHandlerA(ToolHandler[_ProbeInA, _ProbeOutA]):
    invocations: int = 0

    def execute(self, request: ToolExecutionRequest[_ProbeInA]) -> _ProbeOutA:
        _ProbeHandlerA.invocations += 1
        _ = request
        return _ProbeOutA()


class _ProbeHandlerB(ToolHandler[_ProbeInB, _ProbeOutB]):
    invocations: int = 0

    def execute(self, request: ToolExecutionRequest[_ProbeInB]) -> _ProbeOutB:
        _ProbeHandlerB.invocations += 1
        _ = request
        return _ProbeOutB()


def _assert_investigation_policy_provider_messages(messages: list[ChatMessage]) -> None:
    system_contents = [message.content or "" for message in messages if message.role == "system"]
    policy_count = sum(1 for content in system_contents if _INVESTIGATION_POLICY_MARKER in content)
    assert policy_count == 1
    assert not any(_NON_NATIVE_JSON_PROTOCOL_MARKER in content for content in system_contents)
    persistent_system = [
        content
        for content in system_contents
        if _INVESTIGATION_POLICY_MARKER not in content
    ]
    assert not any(_INVESTIGATION_POLICY_MARKER in content for content in persistent_system)


class _InvestigationPolicyThreeRoundLLM(FakeLLMAdapter):
    def __init__(self) -> None:
        super().__init__(fixed_text="")
        self._round = 0

    def supports_tools(self) -> bool:
        return True

    def generate_with_tools(self, messages, tools_schema, **kwargs):  # type: ignore[no-untyped-def]
        _ = tools_schema, kwargs
        self._round += 1
        _assert_investigation_policy_provider_messages(list(messages))

        if self._round == 1:
            return LLMAdapterResponse(
                content="",
                tool_calls=(
                    LLMToolCall.from_openai_shape(
                        call_id="tc-probe-a",
                        name="probe.a",
                        arguments={"query": "status"},
                    ),
                ),
            )

        tool_messages = [message for message in messages if message.role == "tool"]
        tool_contents = [message.content or "" for message in tool_messages]
        if self._round == 2:
            assert len(tool_messages) == 1
            assert any("EVIDENCE_A" in content for content in tool_contents)
            prior_basis = _prior_evidence_references(list(messages))
            return LLMAdapterResponse(
                content="",
                tool_calls=(
                    _action_context_call(*prior_basis, purpose="confirm subgroup from first probe"),
                    LLMToolCall.from_openai_shape(
                        call_id="tc-probe-b",
                        name="probe.b",
                        arguments={"confirm": True},
                    ),
                ),
            )

        assert self._round == 3
        assert len(tool_messages) == 2
        assert any("EVIDENCE_A" in content for content in tool_contents)
        assert any("EVIDENCE_B" in content for content in tool_contents)
        return LLMAdapterResponse(content="investigation complete", tool_calls=())


def test_bounded_react_native_investigation_policy_once_per_provider_call() -> None:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("probe.a", _ProbeInA, _ProbeOutA),
        _ProbeHandlerA(),
    )
    registry.register(
        tools_agent_make_contract("probe.b", _ProbeInB, _ProbeOutB),
        _ProbeHandlerB(),
    )
    _ProbeHandlerA.invocations = 0
    _ProbeHandlerB.invocations = 0
    llm = _InvestigationPolicyThreeRoundLLM()
    state = _runtime_state(llm)
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = ToolPlanningService(llm=llm, tools=registry)
    loop_messages = [ChatMessage(role="user", content="investigate evidence")]

    result = _invoke_bounded_tool_loop(
        state=state,
        invoker=invoker,
        tool_planner=planner,
        planner_input=loop_messages,
        allowed_tool_ids=("probe.a", "probe.b"),
        max_iterations=3,
    )

    assert llm._round == 3
    assert _ProbeHandlerA.invocations == 1
    assert _ProbeHandlerB.invocations == 1
    assert len(result.tool_traces) == 2
    assert result.stop_reason == "planner_final_answer"
    assert not any(
        _INVESTIGATION_POLICY_MARKER in (message.content or "")
        for message in loop_messages
    )
    assert not any(
        _INVESTIGATION_POLICY_MARKER in (message.content or "")
        for message in result.appended_messages
    )


class _NonNativePlanningCaptureLLM(FakeLLMAdapter):
    def __init__(self) -> None:
        super().__init__(fixed_text='{"call_tool": {"name": "alpha.tool", "arguments": {"value": 3}}}')
        self.last_messages: list[ChatMessage] = []

    def supports_tools(self) -> bool:
        return False

    def generate_messages(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        _ = kwargs
        self.last_messages = list(messages)
        return super().generate_messages(messages, **kwargs)


def test_non_native_planning_receives_json_protocol_and_investigation_policy() -> None:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("alpha.tool", _InA, _OutA),
        _HandlerA(),
    )
    llm = _NonNativePlanningCaptureLLM()
    planner = ToolPlanningService(llm=llm, tools=registry)

    decision = planner.plan_tools(
        "use alpha",
        run_id="run-non-native-policy",
        allowed_tool_ids=("alpha.tool",),
    )

    system_contents = [message.content or "" for message in llm.last_messages if message.role == "system"]
    combined = "\n\n".join(system_contents)
    assert _NON_NATIVE_JSON_PROTOCOL_MARKER in combined
    assert '{"call_tool":' in combined
    assert _INVESTIGATION_POLICY_MARKER in combined
    assert decision.tool_plan is not None
    assert len(decision.tool_plan.calls) == 1
    assert decision.tool_plan.calls[0].tool_id == "alpha.tool"
    assert decision.tool_plan.calls[0].input.value == 3
    assert investigation_policy_prompt().strip() in combined


class _EvidenceIn(BaseModel):
    label: str = "probe"


class _EvidenceOut(BaseModel):
    payload: str = "EVIDENCE"


class _EvidenceHandler(ToolHandler[_EvidenceIn, _EvidenceOut]):
    def __init__(self, *, payload: str) -> None:
        self._payload = payload

    def execute(self, request: ToolExecutionRequest[_EvidenceIn]) -> _EvidenceOut:
        _ = request
        return _EvidenceOut(payload=self._payload)


class _MultiHopInvestigationLLM(FakeLLMAdapter):
    def __init__(self) -> None:
        super().__init__(fixed_text="")
        self._round = 0

    def supports_tools(self) -> bool:
        return True

    def generate_with_tools(self, messages, tools_schema, **kwargs):  # type: ignore[no-untyped-def]
        _ = tools_schema, kwargs
        self._round += 1
        _assert_investigation_policy_provider_messages(list(messages))

        if self._round == 1:
            return LLMAdapterResponse(
                content="",
                tool_calls=(
                    LLMToolCall.from_openai_shape(
                        call_id="evidence-a",
                        name="probe.a",
                        arguments={"label": "a"},
                    ),
                ),
            )

        tool_messages = [message for message in messages if message.role == "tool"]
        tool_contents = [message.content or "" for message in tool_messages]

        if self._round == 2:
            assert len(tool_messages) == 1
            assert any("EVIDENCE_A" in content for content in tool_contents)
            prior_basis = _prior_evidence_references(list(messages))
            return LLMAdapterResponse(
                content="",
                tool_calls=(
                    _action_context_call(*prior_basis, purpose="inspect suspected subgroup"),
                    LLMToolCall.from_openai_shape(
                        call_id="evidence-b",
                        name="probe.b",
                        arguments={"label": "b"},
                    ),
                ),
            )

        if self._round == 3:
            assert len(tool_messages) == 2
            assert any("EVIDENCE_A" in content for content in tool_contents)
            assert any("EVIDENCE_B" in content for content in tool_contents)
            prior_basis = _prior_evidence_references(list(messages))
            return LLMAdapterResponse(
                content="",
                tool_calls=(
                    _action_context_call(*prior_basis, purpose="verify normalized effect"),
                    LLMToolCall.from_openai_shape(
                        call_id="evidence-c",
                        name="probe.c",
                        arguments={"label": "c"},
                    ),
                ),
            )

        assert self._round == 4
        assert len(tool_messages) == 3
        return LLMAdapterResponse(content="final investigation answer", tool_calls=())


def test_bounded_react_multi_hop_investigation_proof() -> None:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("probe.a", _EvidenceIn, _EvidenceOut),
        _EvidenceHandler(payload="EVIDENCE_A"),
    )
    registry.register(
        tools_agent_make_contract("probe.b", _EvidenceIn, _EvidenceOut),
        _EvidenceHandler(payload="EVIDENCE_B"),
    )
    registry.register(
        tools_agent_make_contract("probe.c", _EvidenceIn, _EvidenceOut),
        _EvidenceHandler(payload="EVIDENCE_C"),
    )
    llm = _MultiHopInvestigationLLM()
    state = _runtime_state(llm)
    state.context.config.max_tool_iterations = 4
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = ToolPlanningService(llm=llm, tools=registry)

    result = _invoke_bounded_tool_loop(
        state=state,
        invoker=invoker,
        tool_planner=planner,
        planner_input=[ChatMessage(role="user", content="investigate multi-hop")],
        allowed_tool_ids=("probe.a", "probe.b", "probe.c"),
        max_iterations=4,
    )

    proof = result.investigation_proof
    assert proof is not None
    assert result.stop_reason == "planner_final_answer"
    assert llm._round == 4
    assert len(result.tool_traces) == 3
    assert len(proof.steps) == 3

    step1, step2, step3 = proof.steps
    ref_a = f"observation.probe.a.{_INTEGRATION_RUN_ID}:loop1:tool"
    ref_b = f"observation.probe.b.{_INTEGRATION_RUN_ID}:loop2:tool"
    ref_c = f"observation.probe.c.{_INTEGRATION_RUN_ID}:loop3:tool"
    assert step1.round_index == 1
    assert step1.basis_tool_call_ids == ()
    assert step1.next_tool_call_ids == ("evidence-a",)
    assert step2.declared_basis_references == (ref_a,)
    assert step2.basis_tool_call_ids == ("evidence-a",)
    assert step2.next_tool_call_ids == ("evidence-b",)
    assert step2.public_reason == "inspect suspected subgroup"
    assert step3.declared_basis_references == (ref_a, ref_b)
    assert step3.basis_tool_call_ids == ("evidence-a", "evidence-b")
    assert step3.next_tool_call_ids == ("evidence-c",)
    assert step3.public_reason == "verify normalized effect"
    assert proof.final_available_evidence_ids == (
        ref_a,
        ref_b,
        ref_c,
    )


def _probe_business_call() -> LLMToolCall:
    return LLMToolCall.from_openai_shape(
        call_id="evidence-b",
        name="probe.b",
        arguments={"label": "b"},
    )


class _InvalidProofFollowUpLLM(FakeLLMAdapter):
    def __init__(self, *, round2_tool_calls: tuple[LLMToolCall, ...]) -> None:
        super().__init__(fixed_text="")
        self._round = 0
        self._round2_tool_calls = round2_tool_calls

    def supports_tools(self) -> bool:
        return True

    def generate_with_tools(self, messages, tools_schema, **kwargs):  # type: ignore[no-untyped-def]
        _ = tools_schema, kwargs
        self._round += 1
        if self._round == 1:
            return LLMAdapterResponse(
                content="",
                tool_calls=(
                    LLMToolCall.from_openai_shape(
                        call_id="evidence-a",
                        name="probe.a",
                        arguments={"label": "a"},
                    ),
                ),
            )
        return LLMAdapterResponse(content="", tool_calls=self._round2_tool_calls)


def _registry_with_probe_tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("probe.a", _EvidenceIn, _EvidenceOut),
        _EvidenceHandler(payload="EVIDENCE_A"),
    )
    registry.register(
        tools_agent_make_contract("probe.b", _EvidenceIn, _EvidenceOut),
        _EvidenceHandler(payload="EVIDENCE_B"),
    )
    return registry


@pytest.mark.parametrize(
    ("round2_tool_calls", "match"),
    [
        (
            (
                _action_context_call("missing-id", purpose="inspect subgroup"),
                _probe_business_call(),
            ),
            "unknown basis",
        ),
        (
            (_probe_business_call(),),
            "exactly one planner action context",
        ),
        (
            (
                _action_context_call(
                    f"observation.probe.a.{_INTEGRATION_RUN_ID}:loop1:tool",
                    f"observation.probe.a.{_INTEGRATION_RUN_ID}:loop1:tool",
                    purpose="inspect subgroup",
                ),
                _probe_business_call(),
            ),
            "duplicate basis",
        ),
        (
            (
                _action_context_call(purpose="inspect subgroup"),
                _probe_business_call(),
            ),
            "explicit evidence basis",
        ),
    ],
)
def test_investigation_proof_invalid_follow_up_rejected_before_tool_b(
    round2_tool_calls: tuple[LLMToolCall, ...],
    match: str,
) -> None:
    registry = _registry_with_probe_tools()
    llm = _InvalidProofFollowUpLLM(round2_tool_calls=round2_tool_calls)
    state = _runtime_state(llm)
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = ToolPlanningService(llm=llm, tools=registry)

    with pytest.raises(NativePlannerActionContextError, match=match):
        _invoke_bounded_tool_loop(
            state=state,
            invoker=invoker,
            tool_planner=planner,
            planner_input=[ChatMessage(role="user", content="invalid proof")],
            allowed_tool_ids=("probe.a", "probe.b"),
            max_iterations=3,
        )

    assert llm._round == 2
    assert len(state.tool_traces) == 1
    assert state.tool_traces[0].tool_name == "probe.a"


class _OrphanBasisFollowUpLLM(FakeLLMAdapter):
    def __init__(self) -> None:
        super().__init__(fixed_text="")
        self._round = 0

    def supports_tools(self) -> bool:
        return True

    def generate_with_tools(self, messages, tools_schema, **kwargs):  # type: ignore[no-untyped-def]
        _ = messages, tools_schema, kwargs
        self._round += 1
        if self._round == 1:
            return LLMAdapterResponse(
                content="",
                tool_calls=(
                    LLMToolCall.from_openai_shape(
                        call_id="evidence-a",
                        name="probe.a",
                        arguments={"label": "a"},
                    ),
                ),
            )
        return LLMAdapterResponse(
            content="",
            tool_calls=(
                _action_context_call("evidence.orphan.fake", purpose="inspect orphan basis"),
                LLMToolCall.from_openai_shape(
                    call_id="evidence-b",
                    name="probe.b",
                    arguments={"label": "b"},
                ),
            ),
        )


def test_orphan_raw_evidence_basis_rejected_before_second_tool() -> None:
    registry = _registry_with_probe_tools()
    llm = _OrphanBasisFollowUpLLM()
    state = _runtime_state(llm)
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = ToolPlanningService(llm=llm, tools=registry)

    with pytest.raises(NativePlannerActionContextError, match="unknown basis"):
        _invoke_bounded_tool_loop(
            state=state,
            invoker=invoker,
            tool_planner=planner,
            planner_input=[
                ChatMessage(role="tool", content="orphan", tool_call_id="fake-x"),
                ChatMessage(role="user", content="objective"),
            ],
            allowed_tool_ids=("probe.a", "probe.b"),
            max_iterations=3,
        )

    assert llm._round == 2
    assert len(state.tool_traces) == 1
    assert state.tool_traces[0].tool_name == "probe.a"


class _MisalignedCustomPlanner:
    def __init__(self, *, mismatch: str) -> None:
        self._round = 0
        self._mismatch = mismatch

    def plan_tools(
        self,
        input_data,
        context=None,
        *,
        run_id: str,
        allowed_tool_ids=None,
        **kwargs,
    ) -> ToolPlanDecision:
        _ = input_data, context, run_id, allowed_tool_ids, kwargs
        return ToolPlanDecision(
            final_answer=None,
            tool_plan=ToolCallPlan(calls=[]),
            messages=[],
        )

    def plan_native_round(
        self,
        messages,
        *,
        allowed_tool_ids=None,
        run_id=None,
        tool_choice=None,
        **kwargs,
    ):
        _ = messages, allowed_tool_ids, run_id, tool_choice, kwargs
        self._round += 1
        llm_call = LLMToolCall.from_openai_shape(
            call_id="tc-1",
            name="probe.a",
            arguments={"label": "a"},
        )
        llm_call_b = LLMToolCall.from_openai_shape(
            call_id="tc-2",
            name="probe.b",
            arguments={"label": "b"},
        )
        if self._mismatch == "name":
            response = LLMAdapterResponse(content="", tool_calls=(llm_call,))
            tool_plan = ToolCallPlan(
                calls=[
                    PlannedToolCall(
                        step_id="tool",
                        tool_id="probe.b",
                        input=_EvidenceIn(label="a"),
                    )
                ]
            )
            return NativePlannerRound(
                response=response,
                business_tool_calls=(llm_call,),
                tool_plan=tool_plan,
                action_context=None,
            )
        if self._mismatch == "count":
            response = LLMAdapterResponse(content="", tool_calls=(llm_call, llm_call_b))
            tool_plan = ToolCallPlan(
                calls=[
                    PlannedToolCall(
                        step_id="tool",
                        tool_id="probe.a",
                        input=_EvidenceIn(label="a"),
                    )
                ]
            )
            return NativePlannerRound(
                response=response,
                business_tool_calls=(llm_call, llm_call_b),
                tool_plan=tool_plan,
                action_context=None,
            )
        if self._mismatch == "arguments":
            response = LLMAdapterResponse(content="", tool_calls=(llm_call,))
            tool_plan = ToolCallPlan(
                calls=[
                    PlannedToolCall(
                        step_id="tool",
                        tool_id="probe.a",
                        input=_EvidenceIn(label="expected"),
                    )
                ]
            )
            return NativePlannerRound(
                response=response,
                business_tool_calls=(llm_call,),
                tool_plan=tool_plan,
                action_context=None,
            )
        if self._mismatch == "malformed_arguments":
            malformed_call = LLMToolCall(
                id="tc-1",
                name="probe.a",
                arguments_json="{BROKEN",
            )
            response = LLMAdapterResponse(content="", tool_calls=(malformed_call,))
            tool_plan = ToolCallPlan(
                calls=[
                    PlannedToolCall(
                        step_id="tool",
                        tool_id="probe.a",
                        input=_EvidenceIn(),
                    )
                ]
            )
            return NativePlannerRound(
                response=response,
                business_tool_calls=(malformed_call,),
                tool_plan=tool_plan,
                action_context=None,
            )
        raise AssertionError(f"unknown mismatch: {self._mismatch}")


@pytest.mark.parametrize(
    "mismatch",
    ["name", "count", "arguments"],
)
def test_misaligned_custom_planner_rejected_before_tool_execution(
    mismatch: str,
) -> None:
    registry = _registry_with_probe_tools()
    state = _runtime_state(FakeLLMAdapter())
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = _MisalignedCustomPlanner(mismatch=mismatch)

    with pytest.raises(NativeToolPlanAlignmentError):
        _invoke_bounded_tool_loop(
            state=state,
            invoker=invoker,
            tool_planner=planner,
            planner_input=[ChatMessage(role="user", content="misaligned planner")],
            allowed_tool_ids=("probe.a", "probe.b"),
            max_iterations=2,
        )

    assert planner._round == 1
    assert len(state.tool_traces) == 0


def test_malformed_native_arguments_rejected_before_tool_execution() -> None:
    registry = _registry_with_probe_tools()
    state = _runtime_state(FakeLLMAdapter())
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = _MisalignedCustomPlanner(mismatch="malformed_arguments")

    with pytest.raises(NativeToolPlanAlignmentError, match="malformed"):
        _invoke_bounded_tool_loop(
            state=state,
            invoker=invoker,
            tool_planner=planner,
            planner_input=[ChatMessage(role="user", content="malformed native arguments")],
            allowed_tool_ids=("probe.a", "probe.b"),
            max_iterations=2,
        )

    assert planner._round == 1
    assert len(state.tool_traces) == 0
