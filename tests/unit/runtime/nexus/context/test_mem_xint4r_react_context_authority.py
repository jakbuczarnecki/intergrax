# © Artur Czarnecki. All rights reserved.

"""MEM-XINT-4-R — UE-9D ReAct context authority closure."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.context.bootstrap import materialize_context_plugin_registry
from intergrax.context.contracts import (
    AssembledContext,
    ContextAssemblyRequest,
    ContextProviderContext,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine
from intergrax.runtime.nexus.context.iterative_bounded_tool_loop_policy import (
    ContextEngineRequiredForIterativeToolLoopError,
    SyncIterativeBoundedToolLoopForbiddenError,
)
from intergrax.runtime.nexus.context.iterative_tool_context_assembly import (
    run_ce_bounded_tool_loop,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.native_planner_action_context import NativePlannerRound
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.nexus.tools.tool_loop import (
    PlannedToolCallOutcome,
    append_native_tool_messages,
    run_bounded_tool_loop,
    run_bounded_tool_loop_async,
)
from intergrax.runtime.nexus.tracing.trace_models import ToolCallTrace
from intergrax.tools.core.tool_plan import PlannedToolCall, ToolCallPlan
from intergrax.tools.core.tool_plan_decision import ToolPlanDecision
from intergrax.tools.execution_models import ToolExecutionRequest, ToolModelObservation
from intergrax.tools.registry import ToolRegistry
from pydantic import BaseModel
from testing_support.builder import (
    FakeLLMAdapter,
    build_runtime_state_for_tests,
    tools_agent_make_contract,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_BOUNDED_REACT = (
    Path(__file__).resolve().parents[5]
    / "intergrax"
    / "runtime"
    / "nexus"
    / "tools"
    / "patterns"
    / "bounded_react.py"
)


class _In(BaseModel):
    value: int = 1


class _Out(BaseModel):
    result: int = 0


class _Handler:
    def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
        return _Out(result=request.input.value)


class RecordingContextEngine:
    def __init__(self) -> None:
        self.assemble_calls = 0
        self._inner = DefaultNexusContextEngine(
            engine_id="recording",
            registry=materialize_context_plugin_registry(["intergrax.builtin"]),
        )

    @property
    def engine_id(self) -> str:
        return "recording"

    @property
    def registry(self):
        return self._inner.registry

    async def assemble(
        self,
        request: ContextAssemblyRequest,
        *,
        provider_ctx: ContextProviderContext,
    ) -> AssembledContext:
        self.assemble_calls += 1
        return await self._inner.assemble(request, provider_ctx=provider_ctx)


class _TwoToolRoundPlanner:
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
        protocol_config=None,
        **kwargs,
    ) -> NativePlannerRound:
        _ = allowed_tool_ids, run_id, tool_choice, protocol_config, kwargs
        self._round += 1
        if self._round == 1:
            return NativePlannerRound(
                response=LLMAdapterResponse(
                    content="",
                    tool_calls=(
                        LLMToolCall.from_openai_shape(
                            call_id="tc-a",
                            name="probe.read",
                            arguments={"value": 1},
                        ),
                        LLMToolCall.from_openai_shape(
                            call_id="tc-b",
                            name="probe.read",
                            arguments={"value": 2},
                        ),
                    ),
                ),
                materialized_tool_calls=(
                    LLMToolCall.from_openai_shape(
                        call_id="tc-a",
                        name="probe.read",
                        arguments={"value": 1},
                    ),
                    LLMToolCall.from_openai_shape(
                        call_id="tc-b",
                        name="probe.read",
                        arguments={"value": 2},
                    ),
                ),
                tool_plan=ToolCallPlan(
                    calls=[
                        PlannedToolCall(
                            step_id="s-a",
                            tool_id="probe.read",
                            input=_In(value=1),
                        ),
                        PlannedToolCall(
                            step_id="s-b",
                            tool_id="probe.read",
                            input=_In(value=2),
                        ),
                    ]
                ),
                action_context=None,
            )
        self.round_two_messages = list(messages)
        return NativePlannerRound(
            response=LLMAdapterResponse(content="done", tool_calls=()),
            materialized_tool_calls=(),
            tool_plan=ToolCallPlan(calls=[]),
            action_context=None,
        )


class _ThreeHopPlanner:
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
        protocol_config=None,
        **kwargs,
    ) -> NativePlannerRound:
        _ = messages, allowed_tool_ids, run_id, tool_choice, protocol_config, kwargs
        self._round += 1
        if self._round <= 2:
            call_id = f"tc-{self._round}"
            return NativePlannerRound(
                response=LLMAdapterResponse(
                    content="",
                    tool_calls=(
                        LLMToolCall.from_openai_shape(
                            call_id=call_id,
                            name="probe.read",
                            arguments={"value": self._round},
                        ),
                    ),
                ),
                materialized_tool_calls=(
                    LLMToolCall.from_openai_shape(
                        call_id=call_id,
                        name="probe.read",
                        arguments={"value": self._round},
                    ),
                ),
                tool_plan=ToolCallPlan(
                    calls=[
                        PlannedToolCall(
                            step_id=f"s-{self._round}",
                            tool_id="probe.read",
                            input=_In(value=self._round),
                        )
                    ]
                ),
                action_context=None,
            )
        return NativePlannerRound(
            response=LLMAdapterResponse(content="final", tool_calls=()),
            materialized_tool_calls=(),
            tool_plan=ToolCallPlan(calls=[]),
            action_context=None,
        )


def _wire_ce(state) -> RecordingContextEngine:
    engine = RecordingContextEngine()
    state.context.config.llm_adapter = FakeLLMAdapter()
    state.context.config.context_engine = engine
    return engine


def _invoker() -> RuntimeToolInvoker:
    registry = ToolRegistry()
    registry.register(tools_agent_make_contract("probe.read", _In, _Out), _Handler())
    return RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))


def _bind_identity(state) -> tuple[object, object]:
    execution_id = mint_execution_id()
    token = bind_active_execution_identity(
        run_id=state.run_id,
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    budget_token = bind_root_execution_budget(
        execution_id=execution_id,
        ledger=create_execution_budget_ledger(None),
    )
    return token, budget_token


def test_bounded_react_source_forbids_append_native_tool_messages() -> None:
    source = _REPO_BOUNDED_REACT.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_REPO_BOUNDED_REACT))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "append_native_tool_messages"


def test_sync_bounded_tool_loop_rejects_multi_iteration() -> None:
    state = build_runtime_state_for_tests(run_id=mint_run_id())
    with pytest.raises(SyncIterativeBoundedToolLoopForbiddenError):
        run_bounded_tool_loop(
            state=state,
            invoker=_invoker(),
            tool_planner=_TwoToolRoundPlanner(),
            planner_input=[ChatMessage(role="user", content="u")],
            allowed_tool_ids=("probe.read",),
            max_iterations=2,
        )


@pytest.mark.asyncio
async def test_missing_context_engine_fails_closed() -> None:
    state = build_runtime_state_for_tests(run_id=mint_run_id())
    state.context.config.context_engine = None
    with pytest.raises(ContextEngineRequiredForIterativeToolLoopError):
        await run_bounded_tool_loop_async(
            state=state,
            invoker=_invoker(),
            tool_planner=_TwoToolRoundPlanner(),
            planner_input=[ChatMessage(role="user", content="u")],
            allowed_tool_ids=("probe.read",),
            max_iterations=2,
        )


@pytest.mark.asyncio
async def test_canonical_path_does_not_call_append_native_tool_messages() -> None:
    state = build_runtime_state_for_tests(run_id=mint_run_id())
    engine = _wire_ce(state)
    token, budget_token = _bind_identity(state)
    try:
        with patch(
            "intergrax.runtime.nexus.tools.tool_loop.append_native_tool_messages",
            side_effect=AssertionError("append_native_tool_messages must not run"),
        ) as legacy_append:
            result = await run_ce_bounded_tool_loop(
                state=state,
                invoker=_invoker(),
                tool_planner=_TwoToolRoundPlanner(),
                planner_input=[ChatMessage(role="user", content="u")],
                allowed_tool_ids=("probe.read",),
                max_iterations=2,
            )
            legacy_append.assert_not_called()
    finally:
        reset_active_execution_identity(token)
        reset_active_execution_budget(budget_token)

    assert result.used_ce_tool_feedback
    assert engine.assemble_calls >= 2


@pytest.mark.asyncio
async def test_custom_context_engine_used_for_iterative_rounds() -> None:
    state = build_runtime_state_for_tests(run_id=mint_run_id())
    engine = _wire_ce(state)
    planner = _TwoToolRoundPlanner()
    token, budget_token = _bind_identity(state)
    try:
        await run_ce_bounded_tool_loop(
            state=state,
            invoker=_invoker(),
            tool_planner=planner,
            planner_input=[ChatMessage(role="user", content="u")],
            allowed_tool_ids=("probe.read",),
            max_iterations=2,
        )
    finally:
        reset_active_execution_identity(token)
        reset_active_execution_budget(budget_token)

    assert engine.assemble_calls >= 2
    tool_messages = [msg for msg in planner.round_two_messages if msg.role == "tool"]
    assert {msg.tool_call_id for msg in tool_messages} == {"tc-a", "tc-b"}
    assert all(msg.role == "tool" for msg in tool_messages)


@pytest.mark.asyncio
async def test_tool_outcomes_not_duplicated_in_planner_input() -> None:
    state = build_runtime_state_for_tests(run_id=mint_run_id())
    _wire_ce(state)
    planner = _TwoToolRoundPlanner()
    token, budget_token = _bind_identity(state)
    try:
        await run_ce_bounded_tool_loop(
            state=state,
            invoker=_invoker(),
            tool_planner=planner,
            planner_input=[ChatMessage(role="user", content="u")],
            allowed_tool_ids=("probe.read",),
            max_iterations=2,
        )
    finally:
        reset_active_execution_identity(token)
        reset_active_execution_budget(budget_token)

    tool_contents = [msg.content for msg in planner.round_two_messages if msg.role == "tool"]
    assert len(tool_contents) == len(set(tool_contents))


@pytest.mark.asyncio
async def test_multi_hop_preserves_stop_reason_and_investigation_proof() -> None:
    state = build_runtime_state_for_tests(run_id=mint_run_id())
    _wire_ce(state)
    token, budget_token = _bind_identity(state)
    try:
        result = await run_ce_bounded_tool_loop(
            state=state,
            invoker=_invoker(),
            tool_planner=_TwoToolRoundPlanner(),
            planner_input=[ChatMessage(role="user", content="u")],
            allowed_tool_ids=("probe.read",),
            max_iterations=2,
        )
    finally:
        reset_active_execution_identity(token)
        reset_active_execution_budget(budget_token)

    assert result.stop_reason == "planner_final_answer"
    assert result.investigation_proof is not None
    assert len(result.investigation_proof.steps) == 1


_POLICY_MODULE = (
    Path(__file__).resolve().parents[5]
    / "intergrax"
    / "runtime"
    / "nexus"
    / "context"
    / "iterative_bounded_tool_loop_policy.py"
)
_EVIDENCE_GATHERING = (
    Path(__file__).resolve().parents[5]
    / "platform_proofs"
    / "scenarios"
    / "ai_incident_investigation"
    / "application"
    / "evidence_gathering.py"
)
_ORDER_WORKFLOW = (
    Path(__file__).resolve().parents[5]
    / "platform_proofs"
    / "scenarios"
    / "indirect_prompt_injection"
    / "application"
    / "order_workflow.py"
)


def test_iterative_bounded_tool_loop_policy_has_no_runtime_ce_construction() -> None:
    source = _POLICY_MODULE.read_text(encoding="utf-8")
    assert "DefaultNexusContextEngine" not in source
    assert "materialize_context_plugin_registry" not in source
    assert "context_engine =" not in source


def test_platform_proof_react_paths_forbid_asyncio_run_workarounds() -> None:
    for path in (_EVIDENCE_GATHERING, _ORDER_WORKFLOW):
        source = path.read_text(encoding="utf-8")
        assert "asyncio.run" not in source
        assert "run_until_complete" not in source
        assert "new_event_loop" not in source


def test_apply_context_engine_to_runtime_config_preserves_custom_engine() -> None:
    from intergrax.applications._shared.context_wiring import apply_context_engine_to_runtime_config
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
    from intergrax.runtime.nexus.config import RuntimeConfig
    from testing_support.builder import FakeLLMAdapter

    custom = RecordingContextEngine()
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False, context_engine=custom)
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="composition.custom_ce")
    apply_context_engine_to_runtime_config(config, env)
    assert config.context_engine is custom


def test_apply_context_engine_to_runtime_config_materializes_when_unset() -> None:
    from intergrax.applications._shared.context_wiring import apply_context_engine_to_runtime_config
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
    from intergrax.runtime.nexus.config import RuntimeConfig
    from testing_support.builder import FakeLLMAdapter

    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="composition.default_ce")
    apply_context_engine_to_runtime_config(config, env)
    assert config.context_engine is not None


@pytest.mark.asyncio
async def test_run_bounded_tool_loop_async_invokes_context_engine_assemble() -> None:
    state = build_runtime_state_for_tests(run_id=mint_run_id())
    engine = _wire_ce(state)
    token, budget_token = _bind_identity(state)
    try:
        await run_bounded_tool_loop_async(
            state=state,
            invoker=_invoker(),
            tool_planner=_TwoToolRoundPlanner(),
            planner_input=[ChatMessage(role="user", content="u")],
            allowed_tool_ids=("probe.read",),
            max_iterations=2,
        )
    finally:
        reset_active_execution_identity(token)
        reset_active_execution_budget(budget_token)
    assert engine.assemble_calls >= 2


def test_append_native_tool_messages_remains_legacy_helper_only() -> None:
    messages: list[ChatMessage] = [ChatMessage(role="user", content="u")]
    outcome = PlannedToolCallOutcome(
        trace=ToolCallTrace(
            tool_name="probe.read",
            arguments={},
            output_preview="p",
            success=True,
            error_message=None,
            raw_trace={},
        ),
        model_observation=ToolModelObservation(content="legacy"),
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
