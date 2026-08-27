# © Artur Czarnecki. All rights reserved.

"""UE-6C — iterative tool feedback routed through canonical Context Engineering."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import BaseModel

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextFragment,
    ContextFragmentSource,
)
from intergrax.context.bootstrap import materialize_context_plugin_registry
from intergrax.context.providers.legacy_bridge import fragments_from_tool_output_blocks
from intergrax.context.ranker import DefaultContextRanker
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    peek_active_execution_id,
    reset_active_execution_identity,
)
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine
from intergrax.runtime.nexus.context.iterative_tool_context_assembly import (
    assemble_iterative_tool_planner_messages,
    run_ce_bounded_tool_loop,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.nexus.tools.tool_loop import (
    PlannedToolCallOutcome,
    append_native_tool_messages,
    run_bounded_tool_loop_async,
    tool_output_blocks_from_native_round,
)
from intergrax.runtime.nexus.tracing.trace_models import ToolCallTrace
from intergrax.tools.core.tool_plan import PlannedToolCall, ToolCallPlan
from intergrax.tools.core.tool_plan_decision import ToolPlanDecision
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult, ToolModelObservation
from intergrax.tools.registry import ToolRegistry
from testing_support.builder import build_runtime_state_for_tests, tools_agent_make_contract

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_FORBIDDEN_DYNAMIC_TOKENS = frozenset(
    {
        "getattr",
        "setattr",
        "hasattr",
        "inspect",
        "importlib",
        "asyncio.run",
        "run_until_complete",
    }
)

_FULL_OBSERVATION = '{"result": 42, "detail": "full-model-facing-observation"}'
_FAIL_OBSERVATION = "tool failed with model-facing error"


class _In(BaseModel):
    value: int = 1


class _Out(BaseModel):
    result: int = 0


class _WindowAdapter(LLMAdapter):
    provider = "fake"
    model = "fake-window"

    def __init__(self, window: int = 4096) -> None:
        super().__init__()
        self._window = window

    @property
    def context_window_tokens(self) -> int:
        return self._window

    def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:
        _ = messages, kwargs
        return LLMAdapterResponse(content="ok")


def _build_ce_engine(window: int = 4096) -> DefaultNexusContextEngine:
    return DefaultNexusContextEngine(
        engine_id="default",
        registry=materialize_context_plugin_registry(["intergrax.builtin"]),
    )


def _wire_ce_state(
    state: RuntimeState,
    *,
    window: int = 4096,
) -> DefaultNexusContextEngine:
    adapter = _WindowAdapter(window=window)
    engine = _build_ce_engine(window=window)
    state.context.config.llm_adapter = adapter
    state.context.config.context_engine = engine
    state.context.config.max_tool_iterations = 2
    return engine


class _RecordingHandler:
    def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
        return _Out(result=request.input.value)


class _RecordingInvoker(RuntimeToolInvoker):
    def __init__(self) -> None:
        registry = ToolRegistry()
        registry.register(
            tools_agent_make_contract("probe.read", _In, _Out),
            _RecordingHandler(),
        )
        super().__init__(
            registry=registry,
            executor=RegistryToolExecutor(registry),
        )


def _trace_for(step_id: str, tool_name: str) -> ToolCallTrace:
    return ToolCallTrace(
        tool_name=tool_name,
        arguments={"value": 1},
        output_preview="truncated-preview-only",
        success=True,
        error_message=None,
        raw_trace={"step_id": step_id},
    )


def test_tool_output_blocks_convert_to_tool_output_fragments() -> None:
    blocks = tool_output_blocks_from_native_round(
        [
            LLMToolCall.from_openai_shape(
                call_id="tc-1",
                name="probe.read",
                arguments={"value": 1},
            )
        ],
        [PlannedToolCall(step_id="step-1", tool_id="probe.read", input=_In(value=1))],
        [
            PlannedToolCallOutcome(
                trace=_trace_for("step-1", "probe.read"),
                model_observation=ToolModelObservation(content=_FULL_OBSERVATION),
            )
        ],
    )
    fragments = fragments_from_tool_output_blocks(blocks)
    assert fragments and fragments[0].source is ContextFragmentSource.TOOL_OUTPUT
    assert fragments[0].content == _FULL_OBSERVATION
    assert fragments[0].metadata["tool_call_id"] == "tc-1"
    assert fragments[0].metadata["tool_name"] == "probe.read"
    assert fragments[0].metadata["step_id"] == "step-1"


def test_tool_output_block_preserves_full_observation_not_trace_preview() -> None:
    outcome = PlannedToolCallOutcome(
        trace=_trace_for("step-1", "probe.read"),
        model_observation=ToolModelObservation(content=_FULL_OBSERVATION),
    )
    blocks = tool_output_blocks_from_native_round(
        [LLMToolCall.from_openai_shape(call_id="tc-1", name="probe.read", arguments={})],
        [PlannedToolCall(step_id="step-1", tool_id="probe.read", input=_In())],
        [outcome],
    )
    assert blocks[0]["content"] == _FULL_OBSERVATION
    assert blocks[0]["content"] != outcome.trace.output_preview


class _IterativeCePlanner:
    def __init__(self) -> None:
        self._round = 0
        self.round_two_messages: list[ChatMessage] = []

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
        _ = allowed_tool_ids, run_id, tool_choice
        self._round += 1
        if self._round == 1:
            return (
                LLMAdapterResponse(
                    content="round one",
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
        self.round_two_messages = list(messages)
        return (
            LLMAdapterResponse(content="final answer", tool_calls=()),
            ToolCallPlan(calls=[]),
        )


@pytest.mark.asyncio
async def test_iterative_round_two_receives_ce_tool_feedback() -> None:
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    state = build_runtime_state_for_tests(run_id=run_id)
    _wire_ce_state(state)
    invoker = _RecordingInvoker()
    planner = _IterativeCePlanner()

    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    try:
        result = await run_ce_bounded_tool_loop(
            state=state,
            invoker=invoker,
            tool_planner=planner,
            planner_input=[ChatMessage(role="user", content="iterate")],
            allowed_tool_ids=("probe.read",),
            max_iterations=2,
        )
    finally:
        reset_active_execution_identity(token)

    assert result.used_ce_tool_feedback
    assert result.loop_iterations == 2
    tool_messages = [msg for msg in planner.round_two_messages if msg.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "tc-1"
    assert tool_messages[0].content == '{"result":1}'
    assert any(msg.role == "assistant" and msg.tool_calls for msg in planner.round_two_messages)


@pytest.mark.asyncio
async def test_ce_assembly_provenance_includes_tool_output() -> None:
    state = build_runtime_state_for_tests(run_id=mint_run_id())
    engine = _wire_ce_state(state)
    state.iterative_tool_output_blocks = tool_output_blocks_from_native_round(
        [LLMToolCall.from_openai_shape(call_id="tc-prov", name="probe.read", arguments={})],
        [PlannedToolCall(step_id="step-prov", tool_id="probe.read", input=_In())],
        [
            PlannedToolCallOutcome(
                trace=_trace_for("step-prov", "probe.read"),
                model_observation=ToolModelObservation(content=_FULL_OBSERVATION),
            )
        ],
    )
    assembled_messages = await assemble_iterative_tool_planner_messages(
        state,
        engine,
        [ChatMessage(role="user", content="question")],
    )
    assert any(msg.role == "tool" and msg.content == _FULL_OBSERVATION for msg in assembled_messages)


@pytest.mark.asyncio
async def test_failed_tool_observation_reaches_ce() -> None:
    state = build_runtime_state_for_tests(run_id=mint_run_id())
    engine = _wire_ce_state(state)
    state.iterative_tool_output_blocks = [
        {
            "content": _FAIL_OBSERVATION,
            "tool_call_id": "tc-fail",
            "tool_name": "probe.read",
            "step_id": "step-fail",
        }
    ]
    messages = await assemble_iterative_tool_planner_messages(
        state,
        engine,
        [ChatMessage(role="user", content="question")],
    )
    tool_messages = [msg for msg in messages if msg.role == "tool"]
    assert tool_messages and tool_messages[0].content == _FAIL_OBSERVATION


@pytest.mark.asyncio
async def test_multiple_tool_outputs_keep_attribution() -> None:
    blocks = [
        {
            "content": "result-a",
            "tool_call_id": "tc-a",
            "tool_name": "probe.read",
            "step_id": "step-a",
        },
        {
            "content": "result-b",
            "tool_call_id": "tc-b",
            "tool_name": "probe.read",
            "step_id": "step-b",
        },
    ]
    fragments = fragments_from_tool_output_blocks(blocks)
    assert len(fragments) == 2
    assert {fragment.metadata["tool_call_id"] for fragment in fragments} == {"tc-a", "tc-b"}


def test_tool_output_subject_to_ce_inclusion_policy() -> None:
    ranker = DefaultContextRanker()
    request = ContextAssemblyRequest(
        trace_id="t",
        run_id="r",
        task_id="task",
        tenant_id="tenant",
        assembly_scope="acp_step",
        objective="obj",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(),
        assembly_options=TaskContextAssemblyOptions(),
        step_kind="tool_call",
    )
    fragments = [
        ContextFragment(
            fragment_id="tool-low",
            source=ContextFragmentSource.TOOL_OUTPUT,
            source_id="tc-low",
            content="low quality tool output",
            token_estimate=10,
            relevance_score=0.01,
            freshness_score=0.01,
            confidence_score=0.01,
            mandatory=False,
            metadata={"tool_call_id": "tc-low"},
        ),
        ContextFragment(
            fragment_id="tool-high",
            source=ContextFragmentSource.TOOL_OUTPUT,
            source_id="tc-high",
            content="high quality tool output",
            token_estimate=10,
            relevance_score=0.9,
            freshness_score=0.9,
            confidence_score=0.9,
            mandatory=False,
            metadata={"tool_call_id": "tc-high"},
        ),
    ]
    ranked, excluded = ranker.rank_with_exclusions(fragments, request)
    assert ranked and ranked[0].source_id == "tc-high"
    assert excluded and excluded[0][0].source_id == "tc-low"


@pytest.mark.asyncio
async def test_execution_id_stable_across_ce_tool_loop() -> None:
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    state = build_runtime_state_for_tests(run_id=run_id)
    _wire_ce_state(state)
    invoker = _RecordingInvoker()
    planner = _IterativeCePlanner()

    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    try:
        await run_bounded_tool_loop_async(
            state=state,
            invoker=invoker,
            tool_planner=planner,
            planner_input=[ChatMessage(role="user", content="iterate")],
            allowed_tool_ids=("probe.read",),
            max_iterations=2,
        )
        assert peek_active_execution_id() == execution_id
    finally:
        reset_active_execution_identity(token)


def test_certified_ce_path_does_not_call_append_native_tool_messages() -> None:
    source = Path("intergrax/runtime/nexus/context/iterative_tool_context_assembly.py").read_text(
        encoding="utf-8"
    )
    assert "append_native_tool_messages" not in source


def test_certified_ce_path_has_no_forbidden_dynamic_tokens() -> None:
    source = Path("intergrax/runtime/nexus/context/iterative_tool_context_assembly.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not _FORBIDDEN_DYNAMIC_TOKENS.intersection(names)


def test_append_native_tool_messages_remains_legacy_helper() -> None:
    messages: list[ChatMessage] = [ChatMessage(role="user", content="u")]
    outcome = PlannedToolCallOutcome(
        trace=_trace_for("legacy", "probe.read"),
        model_observation=ToolModelObservation(content="legacy-result"),
    )
    append_native_tool_messages(
        messages,
        assistant_content="call",
        tool_calls=[
            LLMToolCall.from_openai_shape(call_id="tc-legacy", name="probe.read", arguments={})
        ],
        outcomes=[outcome],
    )
    assert messages[-1].role == "tool"
    assert messages[-1].content == "legacy-result"
