# © Artur Czarnecki. All rights reserved.

"""Sole typed translation owner: DeclarativePolicyHitlRequiredError → canonical Nexus HITL."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum
from typing import TYPE_CHECKING, Sequence
from uuid import uuid4

from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType, HumanRequest
from intergrax.contracts.declarative_hitl import (
    DeclarativeHitlDecisionPayload,
    DeclarativeHitlPendingApproval,
    DeclarativePolicyHitlSignal,
)
from intergrax.runtime.interrupts.handler import ExecutionInterruptHandler, GovernanceResolution
from intergrax.runtime.nexus.errors.declarative_policy_violation_error import (
    DeclarativePolicyHitlRequiredError,
)
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel
from intergrax.runtime.policy.policy_trace_diagnostics import DeclarativePolicyHitlRequiredDiagV1
from intergrax.tools.execution_models import ToolExecutionRequest

if TYPE_CHECKING:
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState


def generate_invocation_scope_id() -> str:
    return f"dhr_{uuid4().hex}"


def signal_from_error(
    error: DeclarativePolicyHitlRequiredError,
    *,
    state: RuntimeState,
    request: ToolExecutionRequest[object],
    agent_id: str,
    task_id: str,
    policy_provenance_digest: str | None = None,
) -> DeclarativePolicyHitlSignal:
    return DeclarativePolicyHitlSignal(
        invocation_scope_id=generate_invocation_scope_id(),
        task_id=task_id,
        run_id=request.run_id,
        step_id=str(request.step_id),
        tool_id=request.tool_id,
        agent_id=agent_id,
        idempotency_key=request.idempotency_key,
        matched_rule_ids=error.matched_rule_ids,
        policy_provenance_digest=policy_provenance_digest,
        reasons=error.reasons,
    )


def build_human_request(signal: DeclarativePolicyHitlSignal) -> HumanRequest:
    human_request_id = f"hr_{uuid4().hex[:12]}"
    return HumanRequest(
        request_id=human_request_id,
        prompt=(
            f"Declarative policy requires human approval before executing tool "
            f"'{signal.tool_id}'."
        ),
        options=["approve", "reject", "escalate"],
        context_artifacts=[
            f"tool:{signal.tool_id}",
            f"scope:{signal.invocation_scope_id}",
            f"rules:{','.join(signal.matched_rule_ids)}",
        ],
    )


def build_pending_approval(
    signal: DeclarativePolicyHitlSignal,
    *,
    human_request: HumanRequest,
    pause_id: str,
) -> DeclarativeHitlPendingApproval:
    return DeclarativeHitlPendingApproval(
        invocation_scope_id=signal.invocation_scope_id,
        task_id=signal.task_id,
        run_id=signal.run_id,
        step_id=signal.step_id,
        tool_id=signal.tool_id,
        idempotency_key=signal.idempotency_key,
        matched_rule_ids=signal.matched_rule_ids,
        human_request_id=human_request.request_id,
        policy_provenance_digest=signal.policy_provenance_digest,
        agent_id=signal.agent_id,
        pause_id=pause_id,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def build_agent_decision(
    signal: DeclarativePolicyHitlSignal,
    *,
    human_request: HumanRequest,
) -> AgentDecision:
    payload = DeclarativeHitlDecisionPayload.from_signal(signal)
    return AgentDecision(
        type=AgentDecisionType.REQUEST_HUMAN,
        reason="declarative_policy_require_hitl",
        human_request=human_request,
        payload=payload.to_audit_dict(),
    )


@dataclass(frozen=True, slots=True)
class DeclarativePolicyHitlPauseRequired(RuntimeError):
    """Typed pause control-flow exception — not TOOL_ERROR, not ExecutionInterrupt."""

    signal: DeclarativePolicyHitlSignal
    governance: GovernanceResolution
    pending: DeclarativeHitlPendingApproval

    def __str__(self) -> str:
        return (
            f"Declarative policy HITL required for tool '{self.signal.tool_id}' "
            f"(scope={self.signal.invocation_scope_id})."
        )


def _policy_provenance_digest(state: RuntimeState) -> str | None:
    bundle = state.context.config.policy_bundle
    if bundle is None or bundle.declarative_policy_runtime is None:
        return None
    return bundle.declarative_policy_runtime.provenance.rules_digest_sha256


def raise_hitl_pause_from_tool_invocation(
    error: DeclarativePolicyHitlRequiredError,
    *,
    state: RuntimeState,
    request: ToolExecutionRequest[object],
    agent_id: str,
    interrupt_handler: ExecutionInterruptHandler | None = None,
    policy_provenance_digest: str | None = None,
) -> None:
    task_id = state.task_id
    signal = signal_from_error(
        error,
        state=state,
        request=request,
        agent_id=agent_id,
        task_id=task_id,
        policy_provenance_digest=policy_provenance_digest or _policy_provenance_digest(state),
    )
    human_request = build_human_request(signal)
    pause_id = f"pause_{uuid4().hex[:12]}"
    pending = build_pending_approval(signal, human_request=human_request, pause_id=pause_id)
    decision = build_agent_decision(signal, human_request=human_request)
    handler = interrupt_handler or ExecutionInterruptHandler()
    governance = handler.resolve_decision(
        decision,
        task_id=task_id,
        run_id=signal.run_id,
        agent_id=agent_id,
        step_id=signal.step_id,
    )
    state.trace_event(
        component=TraceComponent.TOOLS,
        step="declarative_policy_hitl_required",
        message="Declarative policy requires human approval before tool execution.",
        level=TraceLevel.INFO,
        payload=DeclarativePolicyHitlRequiredDiagV1(
            invocation_scope_id=signal.invocation_scope_id,
            task_id=signal.task_id,
            run_id=signal.run_id,
            step_id=signal.step_id,
            tool_id=signal.tool_id,
            human_request_id=human_request.request_id,
            pause_id=pause_id,
            matched_rule_ids=signal.matched_rule_ids,
        ),
    )
    raise DeclarativePolicyHitlPauseRequired(
        signal=signal,
        governance=governance,
        pending=pending,
    )


class DeclarativeHitlCandidateStatus(StrEnum):
    NO_GRANT = "no_grant"
    UNIQUE = "unique"
    NO_MATCH = "no_match"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class UniqueDeclarativeHitlCandidate:
    candidate_index: int


@dataclass(frozen=True, slots=True)
class DeclarativeHitlScopeCandidateResolution:
    status: DeclarativeHitlCandidateStatus
    candidate_index: int | None = None


@dataclass(frozen=True, slots=True)
class DeclarativeHitlGrantCandidateMismatch(RuntimeError):
    """Fail-closed when grant cannot be uniquely mapped to a ToolExecutionRequest."""

    status: DeclarativeHitlCandidateStatus
    task_id: str

    def __str__(self) -> str:
        return (
            f"Declarative HITL grant candidate resolution failed "
            f"({self.status}, task_id={self.task_id})."
        )


@dataclass
class DeclarativeHitlScopeAssignmentState:
    """Local one-shot guard: grant scope assigned to at most one ToolExecutionRequest."""

    assigned: bool = False


def grant_matches_request_dimensions(
    grant: object,
    request: ToolExecutionRequest[object],
    *,
    task_id: str,
) -> bool:
    from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant

    if not isinstance(grant, DeclarativeHitlApprovalGrant):
        return False
    if grant.task_id != task_id:
        return False
    if grant.run_id != request.run_id:
        return False
    if grant.step_id != str(request.step_id):
        return False
    if grant.tool_id != request.tool_id:
        return False
    if grant.idempotency_key is not None and grant.idempotency_key != request.idempotency_key:
        return False
    return True


def resolve_grant_scope_candidate(
    requests: Sequence[ToolExecutionRequest[object]],
    *,
    grant: object | None,
    task_id: str,
) -> DeclarativeHitlScopeCandidateResolution:
    if grant is None:
        return DeclarativeHitlScopeCandidateResolution(
            status=DeclarativeHitlCandidateStatus.NO_GRANT,
        )
    matches = [
        index
        for index, request in enumerate(requests)
        if grant_matches_request_dimensions(grant, request, task_id=task_id)
    ]
    if len(matches) == 1:
        return DeclarativeHitlScopeCandidateResolution(
            status=DeclarativeHitlCandidateStatus.UNIQUE,
            candidate_index=matches[0],
        )
    if len(matches) == 0:
        return DeclarativeHitlScopeCandidateResolution(
            status=DeclarativeHitlCandidateStatus.NO_MATCH,
        )
    return DeclarativeHitlScopeCandidateResolution(
        status=DeclarativeHitlCandidateStatus.AMBIGUOUS,
    )


def unique_candidate_from_resolution(
    resolution: DeclarativeHitlScopeCandidateResolution,
) -> UniqueDeclarativeHitlCandidate | None:
    if resolution.status is not DeclarativeHitlCandidateStatus.UNIQUE:
        return None
    if resolution.candidate_index is None:
        return None
    return UniqueDeclarativeHitlCandidate(candidate_index=resolution.candidate_index)


def maybe_assign_declarative_hitl_scope(
    request: ToolExecutionRequest[object],
    *,
    state: RuntimeState,
    assignment_state: DeclarativeHitlScopeAssignmentState | None,
    unique_candidate: UniqueDeclarativeHitlCandidate | None = None,
    request_index: int = 0,
) -> ToolExecutionRequest[object]:
    grant = state.declarative_hitl_grant
    if grant is None or assignment_state is None or assignment_state.assigned:
        return request
    if unique_candidate is None or request_index != unique_candidate.candidate_index:
        return request
    task_id = state.task_id
    if not grant_matches_request_dimensions(grant, request, task_id=task_id):
        return request
    assignment_state.assigned = True
    return replace(
        request,
        declarative_hitl_invocation_scope_id=grant.invocation_scope_id,
    )
