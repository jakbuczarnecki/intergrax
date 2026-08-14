# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.agent_decision import AgentDecisionType
from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant
from intergrax.runtime.nexus.errors.declarative_policy_violation_error import (
    DeclarativePolicyHitlRequiredError,
)
from intergrax.runtime.nexus.tools.declarative_policy_hitl_bridge import (
    DeclarativePolicyHitlPauseRequired,
    build_agent_decision,
    build_human_request,
    build_pending_approval,
    generate_invocation_scope_id,
    grant_matches_request_dimensions,
    maybe_assign_declarative_hitl_scope,
    raise_hitl_pause_from_tool_invocation,
    select_grant_scope_candidate_index,
    signal_from_error,
    DeclarativeHitlScopeAssignmentState,
)
from intergrax.tools.execution_models import ToolExecutionRequest
from testing_support.builder import build_runtime_state_for_tests

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _Input:
    value: int = 1


def _error() -> DeclarativePolicyHitlRequiredError:
    return DeclarativePolicyHitlRequiredError(
        run_id="run-1",
        agent_id="agent-1",
        tool_id="tool.a",
        matched_rule_ids=("rule-1",),
        reasons=("needs approval",),
    )


def test_signal_and_pending_scope_created_once() -> None:
    state = build_runtime_state_for_tests(run_id="run-1")
    state.request.metadata["task_id"] = "task-1"
    request = ToolExecutionRequest(
        run_id="run-1",
        step_id="step-1",
        tool_id="tool.a",
        input=_Input(),
        idempotency_key="idem-1",
    )
    scope_a = generate_invocation_scope_id()
    scope_b = generate_invocation_scope_id()
    assert scope_a != scope_b
    assert scope_a.startswith("dhr_")

    signal = signal_from_error(
        _error(),
        state=state,
        request=request,
        agent_id="agent-1",
        task_id="task-1",
        policy_provenance_digest="digest-1",
    )
    human_request = build_human_request(signal)
    pending = build_pending_approval(signal, human_request=human_request, pause_id="pause-1")
    assert pending.invocation_scope_id == signal.invocation_scope_id
    decision = build_agent_decision(signal, human_request=human_request)
    assert decision.type is AgentDecisionType.REQUEST_HUMAN
    assert decision.payload["invocation_scope_id"] == signal.invocation_scope_id


def test_raise_hitl_pause_maps_request_human() -> None:
    state = build_runtime_state_for_tests(run_id="run-1")
    state.request.metadata["task_id"] = "task-1"
    request = ToolExecutionRequest(
        run_id="run-1",
        step_id="step-1",
        tool_id="tool.a",
        input=_Input(),
    )
    with pytest.raises(DeclarativePolicyHitlPauseRequired) as exc_info:
        raise_hitl_pause_from_tool_invocation(
            _error(),
            state=state,
            request=request,
            agent_id="agent-1",
        )
    pause = exc_info.value
    assert pause.governance.should_pause is True
    assert pause.governance.interrupt is None
    assert pause.pending.tool_id == "tool.a"


def test_scope_assignment_one_shot() -> None:
    state = build_runtime_state_for_tests(run_id="run-1")
    state.request.metadata["task_id"] = "task-1"
    grant = DeclarativeHitlApprovalGrant(
        grant_id="grant-1",
        invocation_scope_id="dhr_scope",
        task_id="task-1",
        run_id="run-1",
        step_id="step-1",
        tool_id="tool.a",
        idempotency_key="idem-1",
        matched_rule_ids=("rule-1",),
        human_request_id="hr-1",
        policy_provenance_digest="digest-1",
        pause_id="pause-1",
        approved_at="2026-08-14T00:00:00+00:00",
    )
    state.declarative_hitl_grant = grant
    assignment = DeclarativeHitlScopeAssignmentState()
    req_match = ToolExecutionRequest(
        run_id="run-1",
        step_id="step-1",
        tool_id="tool.a",
        input=_Input(),
        idempotency_key="idem-1",
    )
    req_sibling = ToolExecutionRequest(
        run_id="run-1",
        step_id="step-2",
        tool_id="tool.b",
        input=_Input(),
    )
    scoped = maybe_assign_declarative_hitl_scope(
        req_match,
        state=state,
        assignment_state=assignment,
        candidate_index=0,
        request_index=0,
    )
    assert scoped.declarative_hitl_invocation_scope_id == "dhr_scope"
    sibling = maybe_assign_declarative_hitl_scope(
        req_sibling,
        state=state,
        assignment_state=assignment,
        candidate_index=0,
        request_index=1,
    )
    assert sibling.declarative_hitl_invocation_scope_id is None


def test_candidate_selection_zero_and_multi_fail_closed() -> None:
    grant = DeclarativeHitlApprovalGrant(
        grant_id="grant-1",
        invocation_scope_id="dhr_scope",
        task_id="task-1",
        run_id="run-1",
        step_id="step-1",
        tool_id="tool.a",
        idempotency_key=None,
        matched_rule_ids=("rule-1",),
        human_request_id="hr-1",
        policy_provenance_digest=None,
        pause_id="pause-1",
        approved_at="2026-08-14T00:00:00+00:00",
    )
    requests = [
        ToolExecutionRequest(run_id="run-1", step_id="step-1", tool_id="tool.a", input=_Input()),
        ToolExecutionRequest(run_id="run-1", step_id="step-1", tool_id="tool.a", input=_Input()),
    ]
    assert select_grant_scope_candidate_index(requests, grant=grant, task_id="task-1") is None
    assert select_grant_scope_candidate_index(requests[:1], grant=grant, task_id="task-1") == 0
    assert grant_matches_request_dimensions(grant, requests[0], task_id="task-1") is True
