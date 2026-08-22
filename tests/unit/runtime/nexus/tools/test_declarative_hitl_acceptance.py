# © Artur Czarnecki. All rights reserved.

"""ADR-PLATFORM-PLUGIN-001 acceptance proofs A–P for declarative HITL grants."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.contracts.agent_decision import HumanRequest
from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant
from intergrax.runtime.human.declarative_hitl_grant import DeclarativeHitlGrantCoordinator
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.nexus.errors.declarative_policy_violation_error import (
    DeclarativePolicyHitlRequiredError,
)
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tools.declarative_policy_hitl_bridge import (
    DeclarativeHitlCandidateStatus,
    DeclarativeHitlScopeAssignmentState,
    UniqueDeclarativeHitlCandidate,
    maybe_assign_declarative_hitl_scope,
    resolve_grant_scope_candidate,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.policy.declarative_enforcer import DeclarativePolicyEnforcer
from intergrax.runtime.policy.rules.evaluation import PolicyEvaluationContext
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_contract import TaskPauseRecord
from intergrax.tools.execution_models import ToolExecutionRequest
from testing_support.builder import build_runtime_state_for_tests
from tests.unit.runtime.nexus.tools.conftest import FakeRegistry

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_ACCEPT_TASK_ID = mint_task_id()
_ACCEPT_RUN_ID = mint_run_id()

_TOOL_ID = "accept.hitl.tool"
_RULE_ID = "accept.hitl.tool"


class _Input(BaseModel):
    value: int = 1


class _Output(BaseModel):
    value: int = 0


class _CountingExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
        self.calls += 1
        return _Output(value=request.input.value)


def _grant(**overrides: object) -> DeclarativeHitlApprovalGrant:
    base = {
        "grant_id": "grant-1",
        "invocation_scope_id": "dhr_scope",
        "task_id": _ACCEPT_TASK_ID,
        "run_id": _ACCEPT_RUN_ID,
        "step_id": "step-1",
        "tool_id": _TOOL_ID,
        "idempotency_key": "idem-1",
        "matched_rule_ids": (_RULE_ID,),
        "human_request_id": "hr-1",
        "policy_provenance_digest": "digest-1",
        "pause_id": "pause-1",
        "approved_at": "2026-08-14T00:00:00+00:00",
    }
    base.update(overrides)
    return DeclarativeHitlApprovalGrant(**base)


def _pending() -> object:
    from intergrax.contracts.declarative_hitl import DeclarativeHitlPendingApproval

    return DeclarativeHitlPendingApproval(
        invocation_scope_id="dhr_scope",
        task_id=_ACCEPT_TASK_ID,
        run_id=_ACCEPT_RUN_ID,
        step_id="step-1",
        tool_id=_TOOL_ID,
        idempotency_key="idem-1",
        matched_rule_ids=(_RULE_ID,),
        human_request_id="hr-1",
        policy_provenance_digest="digest-1",
        agent_id="agent-1",
        pause_id="pause-1",
        created_at="2026-08-14T00:00:00+00:00",
    )


def _approve_resolution(task: Task) -> None:
    task.runtime.governance.paused = True
    task.runtime.governance.pause_record = TaskPauseRecord(
        pause_id="pause-1",
        task_id=task.task_id,
        human_request_id="hr-1",
    )
    task.runtime.governance.human_request = HumanRequest(
        request_id="hr-1",
        prompt="approve?",
    )
    HumanPauseCoordinator.resolve_human_response(
        task,
        HumanResponseVerdict.APPROVE,
        approver=local_development_approver_evidence(tenant_id=task.tenant_id),
        pause_id="pause-1",
        human_request_id="hr-1",
    )


def test_a_persisted_grant_absent_after_resume_transfer() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=_ACCEPT_TASK_ID)
    task.runtime.governance.declarative_hitl_pending = _pending()
    _approve_resolution(task)
    DeclarativeHitlGrantCoordinator.create_grant_from_pending(task)
    request = RuntimeRequest(
        agent_id="a1",
        user_id="u1",
        session_id="s1",
        message="x",
        task_id=_ACCEPT_TASK_ID,
        run_id=_ACCEPT_RUN_ID,
    )
    updated = DeclarativeHitlGrantCoordinator.transfer_persisted_grant_for_resume(task, request)
    assert updated.declarative_hitl_grant is not None
    assert task.runtime.governance.declarative_hitl_grant is None


def test_b_typed_in_memory_grant_on_runtime_request() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=_ACCEPT_TASK_ID)
    task.runtime.governance.declarative_hitl_pending = _pending()
    _approve_resolution(task)
    grant = DeclarativeHitlGrantCoordinator.create_grant_from_pending(task)
    request = RuntimeRequest(
        agent_id="a1",
        user_id="u1",
        session_id="s1",
        message="x",
        task_id=_ACCEPT_TASK_ID,
        run_id=_ACCEPT_RUN_ID,
    )
    updated = DeclarativeHitlGrantCoordinator.transfer_persisted_grant_for_resume(task, request)
    state = build_runtime_state_for_tests(run_id=_ACCEPT_RUN_ID)
    state.request = updated
    state.declarative_hitl_grant = updated.declarative_hitl_grant
    assert state.declarative_hitl_grant == grant


def test_c_scope_originates_from_tool_execution_request() -> None:
    state = build_runtime_state_for_tests(run_id=_ACCEPT_RUN_ID)
    state.request.task_id = _ACCEPT_TASK_ID
    state.declarative_hitl_grant = _grant()
    assignment = DeclarativeHitlScopeAssignmentState()
    req = ToolExecutionRequest(
        run_id=_ACCEPT_RUN_ID,
        step_id="step-1",
        tool_id=_TOOL_ID,
        input=_Input(),
        idempotency_key="idem-1",
    )
    scoped = maybe_assign_declarative_hitl_scope(
        req,
        state=state,
        assignment_state=assignment,
        unique_candidate=UniqueDeclarativeHitlCandidate(candidate_index=0),
        request_index=0,
    )
    assert scoped.declarative_hitl_invocation_scope_id == "dhr_scope"
    assert req.declarative_hitl_invocation_scope_id is None


def test_d_different_invocation_scope_cannot_reuse_grant() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="accept.d")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[
            {
                "rule_id": _RULE_ID,
                "handler_id": "deny_tool",
                "resource_kind": "tool",
                "resource_id": _TOOL_ID,
                "action": "require_hitl",
            }
        ],
        policy_enforcement_mode="enforce",
    )
    bundle = wire_policy_bundle(env)
    enforcer = DeclarativePolicyEnforcer(runtime=bundle.declarative_policy_runtime)
    grant = _grant(invocation_scope_id="dhr_scope")
    decision = enforcer.evaluate_tool_invocation(
        context=PolicyEvaluationContext(
            tool_id=_TOOL_ID,
            task_id=_ACCEPT_TASK_ID,
            run_id=_ACCEPT_RUN_ID,
            step_id="step-1",
            idempotency_key="idem-1",
            invocation_scope_id="dhr_other",
            approval_grant=grant,
        )
    )
    assert decision.should_block_execution is True


def test_e_deny_after_approve_does_not_restore_grant() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="accept.e")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[
            {
                "rule_id": _RULE_ID,
                "handler_id": "deny_tool",
                "resource_kind": "tool",
                "resource_id": _TOOL_ID,
                "action": "deny",
            }
        ],
        policy_enforcement_mode="enforce",
    )
    bundle = wire_policy_bundle(env)
    enforcer = DeclarativePolicyEnforcer(runtime=bundle.declarative_policy_runtime)
    grant = _grant()
    decision = enforcer.evaluate_tool_invocation(
        context=PolicyEvaluationContext(
            tool_id=_TOOL_ID,
            task_id=_ACCEPT_TASK_ID,
            run_id=_ACCEPT_RUN_ID,
            step_id="step-1",
            invocation_scope_id="dhr_scope",
            approval_grant=grant,
        )
    )
    assert decision.action.value == "deny"
    assert decision.should_block_execution is True


def test_f_failed_resume_consumes_grant_without_restore() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=_ACCEPT_TASK_ID)
    task.runtime.governance.declarative_hitl_pending = _pending()
    _approve_resolution(task)
    DeclarativeHitlGrantCoordinator.create_grant_from_pending(task)
    request = RuntimeRequest(
        agent_id="a1",
        user_id="u1",
        session_id="s1",
        message="x",
        task_id=_ACCEPT_TASK_ID,
        run_id=_ACCEPT_RUN_ID,
    )
    DeclarativeHitlGrantCoordinator.transfer_persisted_grant_for_resume(task, request)
    assert task.runtime.governance.declarative_hitl_grant is None
    DeclarativeHitlGrantCoordinator.clear_pending_and_grant(task)
    assert task.runtime.governance.declarative_hitl_pending is None


def test_g_multi_tool_candidate_resolution_unique() -> None:
    grant = _grant(idempotency_key=None)
    requests = [
        ToolExecutionRequest(
            run_id=_ACCEPT_RUN_ID,
            step_id="step-1",
            tool_id=_TOOL_ID,
            input=_Input(),
        ),
        ToolExecutionRequest(
            run_id=_ACCEPT_RUN_ID,
            step_id="step-2",
            tool_id="other.tool",
            input=_Input(),
        ),
    ]
    resolution = resolve_grant_scope_candidate(requests, grant=grant, task_id=_ACCEPT_TASK_ID)
    assert resolution.status is DeclarativeHitlCandidateStatus.UNIQUE
    assert resolution.candidate_index == 0


def test_h_approval_targets_exactly_one_call() -> None:
    grant = _grant()
    match = ToolExecutionRequest(
        run_id=_ACCEPT_RUN_ID,
        step_id="step-1",
        tool_id=_TOOL_ID,
        input=_Input(),
        idempotency_key="idem-1",
    )
    resolution = resolve_grant_scope_candidate([match], grant=grant, task_id=_ACCEPT_TASK_ID)
    assert resolution.status is DeclarativeHitlCandidateStatus.UNIQUE


def test_i_only_target_receives_scope() -> None:
    state = build_runtime_state_for_tests(run_id=_ACCEPT_RUN_ID)
    state.request.task_id = _ACCEPT_TASK_ID
    state.declarative_hitl_grant = _grant()
    assignment = DeclarativeHitlScopeAssignmentState()
    target = ToolExecutionRequest(
        run_id=_ACCEPT_RUN_ID,
        step_id="step-1",
        tool_id=_TOOL_ID,
        input=_Input(),
        idempotency_key="idem-1",
    )
    sibling = ToolExecutionRequest(
        run_id=_ACCEPT_RUN_ID,
        step_id="step-2",
        tool_id="other",
        input=_Input(),
    )
    unique = UniqueDeclarativeHitlCandidate(candidate_index=0)
    scoped_target = maybe_assign_declarative_hitl_scope(
        target,
        state=state,
        assignment_state=assignment,
        unique_candidate=unique,
        request_index=0,
    )
    scoped_sibling = maybe_assign_declarative_hitl_scope(
        sibling,
        state=state,
        assignment_state=assignment,
        unique_candidate=unique,
        request_index=1,
    )
    assert scoped_target.declarative_hitl_invocation_scope_id == "dhr_scope"
    assert scoped_sibling.declarative_hitl_invocation_scope_id is None


def test_j_approved_target_executes_once_via_invoker() -> None:
    from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel

    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="accept.j")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[
            {
                "rule_id": _RULE_ID,
                "handler_id": "deny_tool",
                "resource_kind": "tool",
                "resource_id": _TOOL_ID,
                "action": "require_hitl",
            }
        ],
        policy_enforcement_mode="enforce",
    )
    bundle = wire_policy_bundle(env)
    contract = ToolContract(
        tool_id=_TOOL_ID,
        name=_TOOL_ID,
        description="accept",
        input_schema=_Input,
        output_schema=_Output,
        side_effects=False,
        error_mapping={},
        risk_level=ToolRiskLevel.LOW,
    )
    executor = _CountingExecutor()
    invoker = RuntimeToolInvoker(registry=FakeRegistry(contract), executor=executor)
    state = build_runtime_state_for_tests(run_id=_ACCEPT_RUN_ID)
    state.request.task_id = _ACCEPT_TASK_ID
    state.context.config.policy_bundle = bundle
    state.declarative_hitl_grant = _grant(
        matched_rule_ids=(_RULE_ID,),
        policy_provenance_digest=bundle.declarative_policy_runtime.provenance.rules_digest_sha256,
    )
    req = maybe_assign_declarative_hitl_scope(
        ToolExecutionRequest(
            run_id=_ACCEPT_RUN_ID,
            step_id="step-1",
            tool_id=_TOOL_ID,
            input=_Input(),
            idempotency_key="idem-1",
        ),
        state=state,
        assignment_state=DeclarativeHitlScopeAssignmentState(),
        unique_candidate=UniqueDeclarativeHitlCandidate(candidate_index=0),
        request_index=0,
    )
    invoker.invoke(state=state, agent_id="agent-1", request=req)
    assert executor.calls == 1


def test_k_sibling_cannot_reuse_grant_scope() -> None:
    state = build_runtime_state_for_tests(run_id=_ACCEPT_RUN_ID)
    state.request.task_id = _ACCEPT_TASK_ID
    state.declarative_hitl_grant = _grant()
    assignment = DeclarativeHitlScopeAssignmentState()
    unique = UniqueDeclarativeHitlCandidate(candidate_index=0)
    first = maybe_assign_declarative_hitl_scope(
        ToolExecutionRequest(
            run_id=_ACCEPT_RUN_ID,
            step_id="step-1",
            tool_id=_TOOL_ID,
            input=_Input(),
            idempotency_key="idem-1",
        ),
        state=state,
        assignment_state=assignment,
        unique_candidate=unique,
        request_index=0,
    )
    second = maybe_assign_declarative_hitl_scope(
        ToolExecutionRequest(
            run_id=_ACCEPT_RUN_ID,
            step_id="step-1",
            tool_id=_TOOL_ID,
            input=_Input(),
            idempotency_key="idem-2",
        ),
        state=state,
        assignment_state=assignment,
        unique_candidate=unique,
        request_index=1,
    )
    assert first.declarative_hitl_invocation_scope_id == "dhr_scope"
    assert second.declarative_hitl_invocation_scope_id is None


def test_l_repeated_same_tool_id_cannot_reuse_grant() -> None:
    test_k_sibling_cannot_reuse_grant_scope()


def test_m_parallel_sibling_resolution_ambiguous() -> None:
    grant = _grant(idempotency_key=None)
    requests = [
        ToolExecutionRequest(
            run_id=_ACCEPT_RUN_ID,
            step_id="step-1",
            tool_id=_TOOL_ID,
            input=_Input(),
        ),
        ToolExecutionRequest(
            run_id=_ACCEPT_RUN_ID,
            step_id="step-1",
            tool_id=_TOOL_ID,
            input=_Input(),
        ),
    ]
    resolution = resolve_grant_scope_candidate(requests, grant=grant, task_id=_ACCEPT_TASK_ID)
    assert resolution.status is DeclarativeHitlCandidateStatus.AMBIGUOUS


def test_n_zero_candidates_with_grant() -> None:
    grant = _grant()
    resolution = resolve_grant_scope_candidate(
        [
            ToolExecutionRequest(
                run_id=_ACCEPT_RUN_ID,
                step_id="other",
                tool_id="other",
                input=_Input(),
            )
        ],
        grant=grant,
        task_id=_ACCEPT_TASK_ID,
    )
    assert resolution.status is DeclarativeHitlCandidateStatus.NO_MATCH


def test_o_multiple_candidates_with_grant() -> None:
    test_m_parallel_sibling_resolution_ambiguous()


def test_p_missing_task_id_in_context_grant_does_not_satisfy() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="accept.p")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[
            {
                "rule_id": _RULE_ID,
                "handler_id": "deny_tool",
                "resource_kind": "tool",
                "resource_id": _TOOL_ID,
                "action": "require_hitl",
            }
        ],
        policy_enforcement_mode="enforce",
    )
    bundle = wire_policy_bundle(env)
    enforcer = DeclarativePolicyEnforcer(runtime=bundle.declarative_policy_runtime)
    grant = _grant()
    decision = enforcer.evaluate_tool_invocation(
        context=PolicyEvaluationContext(
            tool_id=_TOOL_ID,
            run_id=_ACCEPT_RUN_ID,
            step_id="step-1",
            invocation_scope_id="dhr_scope",
            approval_grant=grant,
        )
    )
    assert decision.should_block_execution is True


def test_invoker_blocks_without_scope_when_hitl_required() -> None:
    from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel

    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="accept.invoker")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[
            {
                "rule_id": _RULE_ID,
                "handler_id": "deny_tool",
                "resource_kind": "tool",
                "resource_id": _TOOL_ID,
                "action": "require_hitl",
            }
        ],
        policy_enforcement_mode="enforce",
    )
    bundle = wire_policy_bundle(env)
    contract = ToolContract(
        tool_id=_TOOL_ID,
        name=_TOOL_ID,
        description="accept",
        input_schema=_Input,
        output_schema=_Output,
        side_effects=False,
        error_mapping={},
        risk_level=ToolRiskLevel.LOW,
    )
    executor = _CountingExecutor()
    invoker = RuntimeToolInvoker(registry=FakeRegistry(contract), executor=executor)
    state = build_runtime_state_for_tests(run_id=_ACCEPT_RUN_ID)
    state.request.task_id = _ACCEPT_TASK_ID
    state.context.config.policy_bundle = bundle
    with pytest.raises(DeclarativePolicyHitlRequiredError):
        invoker.invoke(
            state=state,
            agent_id="agent-1",
            request=ToolExecutionRequest(
                run_id=_ACCEPT_RUN_ID,
                step_id="step-1",
                tool_id=_TOOL_ID,
                input=_Input(),
            ),
        )
    assert executor.calls == 0
