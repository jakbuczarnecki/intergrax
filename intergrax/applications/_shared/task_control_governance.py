# © Artur Czarnecki. All rights reserved.

"""Control-plane mutation helpers for governed task-control cancel/autonomy (CLA-CPM)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.autonomy_level import AutonomyLevel
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationAuthorizationEvidence,
    ControlPlaneMutationAuthorizationResult,
    ControlPlaneMutationAuthorizationScope,
    ControlPlaneMutationRequest,
    ControlPlaneMutationRisk,
)
from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_contract import TaskPauseRecord

TASK_EXECUTION_RESOURCE_TYPE = "task_control.task_execution"
MUTATION_TYPE_CANCEL_TASK_EXECUTION = "task_control.cancel_task_execution"
MUTATION_TYPE_SET_TASK_AUTONOMY = "task_control.set_task_autonomy"
MUTATION_TYPE_RESUME_TASK_EXECUTION = "task_control.resume_task_execution"

_TERMINAL_TASK_STATES = frozenset(
    {
        TaskState.COMPLETED,
        TaskState.FAILED,
        TaskState.CANCELLED,
        TaskState.EXPIRED,
        TaskState.PARTIALLY_COMPLETED,
    }
)


class TaskControlTenantScopeDenial(BaseModel):
    """Pre-evaluation tenant authority rejection — no mutation request was evaluated."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1)
    resource_type: str = Field(min_length=1)
    resource_id: str = Field(min_length=1)
    resource_scope: str = Field(min_length=1)
    principal_type: PrincipalType
    principal_user_id: str | None = None
    principal_auth_subject: str | None = None
    reason: str = Field(min_length=1)


class TaskControlGovernanceBlockedError(Exception):
    """Control-plane task-control mutation blocked before cooperative cancellation."""

    def __init__(
        self,
        blocker_code: str,
        message: str,
        *,
        policy_action: str,
        authorization_evidence: ControlPlaneMutationAuthorizationEvidence | None = None,
        authorization_scope: ControlPlaneMutationAuthorizationScope | None = None,
        tenant_scope_denial: TaskControlTenantScopeDenial | None = None,
    ) -> None:
        super().__init__(message)
        self.blocker_code = blocker_code
        self.policy_action = policy_action
        self.authorization_evidence = authorization_evidence
        self.authorization_scope = authorization_scope
        self.tenant_scope_denial = tenant_scope_denial

    def governance_http_detail(self) -> dict[str, object]:
        detail: dict[str, object] = {
            "blocker_code": self.blocker_code,
            "policy_action": self.policy_action,
        }
        if self.authorization_evidence is not None:
            detail["authorization_evidence"] = self.authorization_evidence.model_dump(
                mode="json"
            )
        if self.authorization_scope is not None:
            detail["authorization_scope"] = self.authorization_scope.model_dump(
                mode="json"
            )
        if self.tenant_scope_denial is not None:
            detail["tenant_scope_denial"] = self.tenant_scope_denial.model_dump(
                mode="json"
            )
        return detail


def task_execution_resource_id(*, task_id: TaskId | str, run_id: RunId | str) -> str:
    return f"{task_id}:{run_id}"


def task_execution_resource_scope(
    *,
    tenant_id: str,
    task_id: TaskId | str,
    run_id: RunId | str,
) -> str:
    return f"task_control.tenant:{tenant_id}.task:{task_id}.run:{run_id}"


def task_execution_state_revision(*, state: str) -> str:
    normalized = state.strip()
    if not normalized:
        raise ValueError("task execution state revision must be non-empty")
    return f"state:{normalized}"


def cancel_requested_target_revision() -> str:
    return task_execution_state_revision(state="cancel_requested")


def task_execution_autonomy_revision(*, autonomy_level: AutonomyLevel | None) -> str:
    if autonomy_level is None:
        return "autonomy:unset"
    return f"autonomy:{autonomy_level.value}"


def _pause_id_from_checkpoint(checkpoint: TaskCheckpoint) -> str | None:
    snapshot = Task.model_validate(checkpoint.task_snapshot)
    pause_record: TaskPauseRecord | None = snapshot.runtime.governance.pause_record
    if pause_record is None:
        return None
    return pause_record.pause_id


def task_checkpoint_stable_identity(checkpoint: TaskCheckpoint) -> str:
    pause_id = _pause_id_from_checkpoint(checkpoint) or "none"
    return (
        f"{checkpoint.checkpoint_id}:{checkpoint.resume_token}:"
        f"{checkpoint.task_state.value}:{pause_id}"
    )


def task_checkpoint_resume_current_revision(checkpoint: TaskCheckpoint) -> str:
    return f"checkpoint:{task_checkpoint_stable_identity(checkpoint)}:paused"


def task_checkpoint_resume_requested_target_revision(checkpoint: TaskCheckpoint) -> str:
    return f"checkpoint:{task_checkpoint_stable_identity(checkpoint)}:resume_requested"


def is_task_execution_cancellable(*, state: TaskState, cancellation_requested: bool) -> bool:
    if state in _TERMINAL_TASK_STATES:
        return False
    return not cancellation_requested


def build_cancel_task_execution_mutation_request(
    *,
    principal: RequestIdentity,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    mutation_id: str,
    current_state: TaskState,
    approval_evidence_ref: str | None = None,
) -> ControlPlaneMutationRequest:
    return ControlPlaneMutationRequest(
        mutation_id=mutation_id,
        mutation_type=MUTATION_TYPE_CANCEL_TASK_EXECUTION,
        principal=principal,
        resource_scope=task_execution_resource_scope(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
        ),
        resource_type=TASK_EXECUTION_RESOURCE_TYPE,
        resource_id=task_execution_resource_id(task_id=task_id, run_id=run_id),
        current_revision=task_execution_state_revision(state=current_state.value),
        target_revision=cancel_requested_target_revision(),
        risk_classification=ControlPlaneMutationRisk.MEDIUM,
        approval_evidence_ref=approval_evidence_ref,
        task_id=task_id,
        run_id=run_id,
    )


def build_resume_task_execution_mutation_request(
    *,
    principal: RequestIdentity,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    mutation_id: str,
    checkpoint: TaskCheckpoint,
    approval_evidence_ref: str | None = None,
) -> ControlPlaneMutationRequest:
    return ControlPlaneMutationRequest(
        mutation_id=mutation_id,
        mutation_type=MUTATION_TYPE_RESUME_TASK_EXECUTION,
        principal=principal,
        resource_scope=task_execution_resource_scope(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
        ),
        resource_type=TASK_EXECUTION_RESOURCE_TYPE,
        resource_id=task_execution_resource_id(task_id=task_id, run_id=run_id),
        current_revision=task_checkpoint_resume_current_revision(checkpoint),
        target_revision=task_checkpoint_resume_requested_target_revision(checkpoint),
        risk_classification=ControlPlaneMutationRisk.MEDIUM,
        approval_evidence_ref=approval_evidence_ref,
        task_id=task_id,
        run_id=run_id,
    )


def build_set_task_autonomy_mutation_request(
    *,
    principal: RequestIdentity,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    mutation_id: str,
    current_autonomy_level: AutonomyLevel | None,
    target_autonomy_level: AutonomyLevel,
    approval_evidence_ref: str | None = None,
) -> ControlPlaneMutationRequest:
    return ControlPlaneMutationRequest(
        mutation_id=mutation_id,
        mutation_type=MUTATION_TYPE_SET_TASK_AUTONOMY,
        principal=principal,
        resource_scope=task_execution_resource_scope(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
        ),
        resource_type=TASK_EXECUTION_RESOURCE_TYPE,
        resource_id=task_execution_resource_id(task_id=task_id, run_id=run_id),
        current_revision=task_execution_autonomy_revision(autonomy_level=current_autonomy_level),
        target_revision=task_execution_autonomy_revision(autonomy_level=target_autonomy_level),
        risk_classification=ControlPlaneMutationRisk.MEDIUM,
        approval_evidence_ref=approval_evidence_ref,
        task_id=task_id,
        run_id=run_id,
    )


def validate_task_control_principal_tenant_authority(
    *,
    principal: RequestIdentity,
    task_tenant_id: str,
    task_id: TaskId | str,
    run_id: RunId | str,
    operation: str,
) -> None:
    if principal.tenant_id != task_tenant_id:
        resource_scope = task_execution_resource_scope(
            tenant_id=task_tenant_id,
            task_id=task_id,
            run_id=run_id,
        )
        resource_id = task_execution_resource_id(task_id=task_id, run_id=run_id)
        raise TaskControlGovernanceBlockedError(
            "TASK_CONTROL_BLOCKED_BY_TENANT_AUTHORITY",
            f"{operation} denied by tenant authority scope",
            policy_action=PolicyAction.DENY.value,
            tenant_scope_denial=TaskControlTenantScopeDenial(
                tenant_id=task_tenant_id,
                resource_type=TASK_EXECUTION_RESOURCE_TYPE,
                resource_id=resource_id,
                resource_scope=resource_scope,
                principal_type=principal.principal_type,
                principal_user_id=principal.user_id,
                principal_auth_subject=principal.auth_subject,
                reason="principal_tenant_mismatch",
            ),
        )


def enforce_task_control_authorization_result(
    result: ControlPlaneMutationAuthorizationResult,
    *,
    operation: str,
) -> ControlPlaneMutationAuthorizationResult:
    if result.permitted:
        return result
    action = result.decision.action
    if action is PolicyAction.REQUIRE_HUMAN:
        raise TaskControlGovernanceBlockedError(
            "TASK_CONTROL_BLOCKED_BY_REQUIRE_HUMAN",
            f"{operation} requires governed human approval",
            policy_action=action.value,
            authorization_evidence=result.evidence,
            authorization_scope=result.authorization_scope,
        )
    if action is PolicyAction.ESCALATE:
        raise TaskControlGovernanceBlockedError(
            "TASK_CONTROL_BLOCKED_BY_ESCALATE",
            f"{operation} requires escalation",
            policy_action=action.value,
            authorization_evidence=result.evidence,
            authorization_scope=result.authorization_scope,
        )
    raise TaskControlGovernanceBlockedError(
        "TASK_CONTROL_BLOCKED_BY_POLICY",
        f"{operation} denied by control-plane policy",
        policy_action=action.value,
        authorization_evidence=result.evidence,
        authorization_scope=result.authorization_scope,
    )
