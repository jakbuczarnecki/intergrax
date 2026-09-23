# © Artur Czarnecki. All rights reserved.

"""EE L3 bridge: Agent Governance REQUIRE_APPROVAL → canonical human pause (R5.5)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

from intergrax.contracts.agent_decision import HumanRequest
from intergrax.contracts.agent_governance_approval_pause_signal import (
    AgentGovernanceApprovalPauseSignal,
)
from intergrax.contracts.agent_governance_hitl import (
    AgentGovernanceHumanApprovalPending,
    AgentGovernanceHumanApprovalRequirement,
    LogicalInvocationFingerprint,
    mint_agent_governance_invocation_scope_id,
)
from intergrax.contracts.agent_runtime_governance import (
    PolicyEvaluationResult,
    ToolAuthorizationRequest,
)
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.runtime.agent_governance.errors import (
    ToolGovernanceApprovalRequiredError,
)
from intergrax.runtime.agent_governance.request_builder import (
    build_tool_authorization_request,
)
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel
from intergrax.runtime.nexus.tracing.tools.tool_invocation import (
    ToolInvocationErrorDiagV1,
)
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.task.task import Task
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest

if TYPE_CHECKING:
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState


@dataclass(frozen=True, slots=True)
class AgentGovernanceApprovalPauseRequired(RuntimeError):
    """Typed pause control-flow for Agent Governance — not TOOL_ERROR."""

    signal: AgentGovernanceApprovalPauseSignal

    def __str__(self) -> str:
        return (
            f"Agent runtime governance requires human approval for tool "
            f"'{self.signal.tool_id}'."
        )


def translate_agent_governance_approval_error(
    error: ToolGovernanceApprovalRequiredError,
    *,
    authorization_request: ToolAuthorizationRequest,
    execution_id: str,
    step_id: str,
    idempotency_key: str | None,
) -> AgentGovernanceApprovalPauseSignal:
    if error.governed_continuation_request is not None:
        raise error
    return AgentGovernanceApprovalPauseSignal(
        task_id=authorization_request.task_id,
        run_id=authorization_request.run_id,
        attempt_id=authorization_request.attempt_id,
        execution_id=execution_id,
        tenant_id=authorization_request.agent.tenant_id,
        agent_id=error.agent_id,
        tool_id=error.tool_id,
        step_id=step_id,
        capability=error.capability,
        idempotency_key=idempotency_key,
        approval_id=error.approval_id,
        reason=error.reason,
        policy_results=error.policy_results,
        policy_provenance_digest=digest_agent_governance_policy_provenance(
            error.policy_results,
        ),
        authorization_request=authorization_request,
    )


def digest_agent_governance_policy_provenance(
    policy_results: tuple[PolicyEvaluationResult, ...],
) -> str | None:
    if not policy_results:
        return None
    parts = tuple(
        f"{item.policy_id}:{item.decision.value}:{item.reason}"
        for item in policy_results
    )
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def build_human_request_for_agent_governance_pause(
    *,
    scope_id: str,
    tool_id: str,
    capability: str,
    reason: str,
) -> HumanRequest:
    human_request_id = f"hr_{uuid4().hex[:12]}"
    return HumanRequest(
        request_id=human_request_id,
        prompt=(
            f"Agent runtime governance requires human approval before executing "
            f"tool '{tool_id}' ({capability}): {reason}"
        ),
        options=[
            HumanResponseVerdict.APPROVE.value,
            HumanResponseVerdict.REJECT.value,
            HumanResponseVerdict.ESCALATE.value,
        ],
        context_artifacts=[
            f"tool:{tool_id}",
            f"capability:{capability}",
            f"scope:{scope_id}",
        ],
    )


def build_agent_governance_pause_artifacts(
    signal: AgentGovernanceApprovalPauseSignal,
    *,
    logical_invocation_fingerprint: LogicalInvocationFingerprint,
    payload_digest: str,
    invocation_scope_id: str | None = None,
    pause_id: str | None = None,
    human_request_id: str | None = None,
) -> tuple[
    AgentGovernanceHumanApprovalRequirement,
    AgentGovernanceHumanApprovalPending,
    HumanRequest,
]:
    scope_id = invocation_scope_id or mint_agent_governance_invocation_scope_id()
    if not scope_id.startswith("agr_"):
        raise ValueError("agent governance invocation scope must use agr_ prefix")
    requirement = AgentGovernanceHumanApprovalRequirement(
        agent_governance_invocation_scope_id=scope_id,
        task_id=signal.task_id,
        run_id=signal.run_id,
        attempt_id=signal.attempt_id,
        execution_id=signal.execution_id,
        tenant_id=signal.tenant_id,
        agent_id=signal.agent_id,
        tool_id=signal.tool_id,
        step_id=signal.step_id,
        idempotency_key=signal.idempotency_key,
        approval_id=signal.approval_id,
        authorization_request=signal.authorization_request,
        policy_results=signal.policy_results,
        policy_provenance_digest=signal.policy_provenance_digest,
        logical_invocation_fingerprint=logical_invocation_fingerprint,
        pause_generation=1,
    )
    resolved_human_request_id = human_request_id or f"hr_{uuid4().hex[:12]}"
    human_request = build_human_request_for_agent_governance_pause(
        scope_id=scope_id,
        tool_id=signal.tool_id,
        capability=signal.capability,
        reason=signal.reason,
    )
    human_request = human_request.model_copy(
        update={"request_id": resolved_human_request_id},
    )
    resolved_pause_id = pause_id or f"pause_{uuid4().hex[:12]}"
    pending = AgentGovernanceHumanApprovalPending(
        agent_governance_invocation_scope_id=scope_id,
        requirement=requirement,
        task_id=signal.task_id,
        run_id=signal.run_id,
        attempt_id=signal.attempt_id,
        execution_id=signal.execution_id,
        tenant_id=signal.tenant_id,
        agent_id=signal.agent_id,
        tool_id=signal.tool_id,
        step_id=signal.step_id,
        idempotency_key=signal.idempotency_key,
        human_request_id=human_request.request_id,
        pause_id=resolved_pause_id,
        policy_provenance_digest=signal.policy_provenance_digest,
        created_at=datetime.now(timezone.utc).isoformat(),
        generation=1,
    )
    return requirement, pending, human_request


def assert_agent_governance_pause_identity_consistency(
    signal: AgentGovernanceApprovalPauseSignal,
    *,
    request: ExecutionBoundCatalogToolInvokeRequest,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId,
) -> None:
    if str(signal.task_id) != str(validate_task_id(request.task_id)):
        raise RuntimeError("agent governance pause task_id mismatch")
    if str(signal.run_id) != str(validate_run_id(run_id)):
        raise RuntimeError("agent governance pause run_id mismatch")
    if str(signal.attempt_id) != str(validate_attempt_id(attempt_id)):
        raise RuntimeError("agent governance pause attempt_id mismatch")
    if str(signal.execution_id) != str(validate_execution_id(execution_id)):
        raise RuntimeError("agent governance pause execution_id mismatch")
    if signal.tenant_id != request.tenant_id:
        raise RuntimeError("agent governance pause tenant_id mismatch")
    if signal.agent_id != request.agent_id:
        raise RuntimeError("agent governance pause agent_id mismatch")
    if signal.tool_id != request.tool_id:
        raise RuntimeError("agent governance pause tool_id mismatch")
    if signal.step_id != str(request.step_id):
        raise RuntimeError("agent governance pause step_id mismatch")


def project_agent_governance_pause_onto_task(
    task: Task,
    *,
    pending: AgentGovernanceHumanApprovalPending,
    human_request: HumanRequest,
) -> None:
    raise RuntimeError(
        "agent governance pause projection requires TaskCheckpointPersistence CAS",
    )


def raise_agent_governance_pause_from_tool_invocation(
    error: ToolGovernanceApprovalRequiredError,
    *,
    state: RuntimeState,
    contract: ToolContract,
    request: ToolExecutionRequest[object],
    agent_id: str,
) -> None:
    if error.governed_continuation_request is not None:
        raise error
    authorization_request = build_tool_authorization_request(
        state=state,
        agent_id=agent_id,
        contract=contract,
        request=request,
    )
    from intergrax.contracts.execution_identity import require_active_execution_id

    execution_id = str(require_active_execution_id())
    signal = translate_agent_governance_approval_error(
        error,
        authorization_request=authorization_request,
        execution_id=execution_id,
        step_id=str(request.step_id),
        idempotency_key=request.idempotency_key,
    )
    state.trace_event(
        component=TraceComponent.TOOLS,
        step="agent_governance_approval_pause_required",
        message="Agent runtime governance requires canonical human pause.",
        level=TraceLevel.INFO,
        payload=ToolInvocationErrorDiagV1(
            tool_id=request.tool_id,
            step_id=str(request.step_id),
            error_code=RuntimeErrorCode.PERMISSION_ERROR,
            error_message=str(error),
        ),
    )
    raise AgentGovernanceApprovalPauseRequired(signal=signal)


__all__ = [
    "AgentGovernanceApprovalPauseRequired",
    "assert_agent_governance_pause_identity_consistency",
    "build_agent_governance_pause_artifacts",
    "digest_agent_governance_policy_provenance",
    "project_agent_governance_pause_onto_task",
    "raise_agent_governance_pause_from_tool_invocation",
    "translate_agent_governance_approval_error",
]
