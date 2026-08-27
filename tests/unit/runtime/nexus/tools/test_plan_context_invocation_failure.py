# © Artur Czarnecki. All rights reserved.

"""ENG-2 — run_tools_context failure propagation."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.errors.declarative_policy_violation_error import (
    DeclarativePolicyHitlRequiredError,
)
from intergrax.runtime.nexus.tools.declarative_policy_hitl_bridge import (
    DeclarativePolicyHitlPauseRequired,
    raise_hitl_pause_from_tool_invocation,
)
from intergrax.runtime.nexus.tools.plan_context_invocation import run_tools_context
from intergrax.runtime.nexus.tracing.tools.tools_summary import ToolsSummaryDiagV1
from intergrax.runtime.nexus.tracing.trace_models import TraceLevel
from intergrax.tools.execution_models import ToolExecutionRequest
from testing_support.builder import build_runtime_state_for_tests

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _ToolLoopExplodedError(RuntimeError):
    pass


@pytest.mark.asyncio
async def test_run_tools_context_emits_tools_summary_then_propagates_failure() -> None:
    state = build_runtime_state_for_tests(run_id="run_tools_ctx_fail")
    state.context.config.tools_mode = "auto"
    state.context.config.tool_invoker = object()
    state.context.config.tool_planner = object()
    state.messages_for_llm = [ChatMessage(role="user", content="use tool")]
    planner_input = [ChatMessage(role="user", content="use tool")]

    with (
        patch(
            "intergrax.runtime.nexus.tools.plan_context_invocation.resolve_tool_planner_input",
            return_value=planner_input,
        ),
        patch(
            "intergrax.runtime.nexus.tools.plan_context_invocation.resolve_tool_registry",
            return_value=None,
        ),
        patch(
            "intergrax.runtime.nexus.tools.plan_context_invocation.run_bounded_tool_loop_async",
            side_effect=_ToolLoopExplodedError("tool loop exploded deterministically"),
        ),
        pytest.raises(_ToolLoopExplodedError, match="tool loop exploded deterministically"),
    ):
        await run_tools_context(state)

    tools_events = [event for event in state.trace_events if event.step == "tools"]
    assert len(tools_events) == 1
    assert tools_events[0].level is TraceLevel.ERROR
    payload = tools_events[0].payload
    assert isinstance(payload, ToolsSummaryDiagV1)
    assert payload.error_type == "_ToolLoopExplodedError"
    assert payload.error_message is not None
    assert payload.used_tools is False


class _MetadataMergeExplodedError(RuntimeError):
    pass


@pytest.mark.asyncio
async def test_run_tools_context_failure_reraises_before_metadata_merge() -> None:
    state = build_runtime_state_for_tests(run_id="run_tools_ctx_metadata_order")
    state.context.config.tools_mode = "auto"
    state.context.config.tool_invoker = object()
    state.context.config.tool_planner = object()
    state.messages_for_llm = [ChatMessage(role="user", content="use tool")]
    planner_input = [ChatMessage(role="user", content="use tool")]

    with (
        patch(
            "intergrax.runtime.nexus.tools.plan_context_invocation.resolve_tool_planner_input",
            return_value=planner_input,
        ),
        patch(
            "intergrax.runtime.nexus.tools.plan_context_invocation.resolve_tool_registry",
            return_value=None,
        ),
        patch(
            "intergrax.runtime.nexus.tools.plan_context_invocation.run_bounded_tool_loop_async",
            side_effect=_ToolLoopExplodedError("tool loop exploded deterministically"),
        ),
        patch(
            "intergrax.runtime.nexus.tools.plan_context_invocation.merge_provider_metadata_into_request",
            side_effect=_MetadataMergeExplodedError("metadata merge must not run on failure"),
        ) as merge_mock,
        pytest.raises(_ToolLoopExplodedError, match="tool loop exploded deterministically"),
    ):
        await run_tools_context(state)

    merge_mock.assert_not_called()
    tools_events = [event for event in state.trace_events if event.step == "tools"]
    assert len(tools_events) == 1
    payload = tools_events[0].payload
    assert isinstance(payload, ToolsSummaryDiagV1)
    assert payload.error_type == "_ToolLoopExplodedError"


def _hitl_error(*, run_id: str) -> DeclarativePolicyHitlRequiredError:
    return DeclarativePolicyHitlRequiredError(
        run_id=run_id,
        agent_id="agent_test",
        tool_id="tool.a",
        matched_rule_ids=("rule-1",),
        reasons=("needs approval",),
    )


@pytest.mark.asyncio
async def test_run_tools_context_hitl_pause_propagates_without_tools_summary() -> None:
    state = build_runtime_state_for_tests(run_id="run_tools_ctx_hitl")
    state.context.config.tools_mode = "auto"
    state.context.config.tool_invoker = object()
    state.context.config.tool_planner = object()
    planner_input = [ChatMessage(role="user", content="use tool")]

    def _raise_hitl_pause(*, state, **_kwargs):  # type: ignore[no-untyped-def]
        request = ToolExecutionRequest(
            run_id=state.run_id,
            step_id="step-1",
            tool_id="tool.a",
            input=object(),
        )
        raise_hitl_pause_from_tool_invocation(
            _hitl_error(run_id=state.run_id),
            state=state,
            request=request,
            agent_id=state.request.agent_id,
        )

    with (
        patch(
            "intergrax.runtime.nexus.tools.plan_context_invocation.resolve_tool_planner_input",
            return_value=planner_input,
        ),
        patch(
            "intergrax.runtime.nexus.tools.plan_context_invocation.resolve_tool_registry",
            return_value=None,
        ),
        patch(
            "intergrax.runtime.nexus.tools.plan_context_invocation.run_bounded_tool_loop_async",
            side_effect=_raise_hitl_pause,
        ),
        pytest.raises(DeclarativePolicyHitlPauseRequired),
    ):
        await run_tools_context(state)

    assert not any(event.step == "tools" for event in state.trace_events)
