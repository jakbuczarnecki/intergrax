# © Artur Czarnecki. All rights reserved.

"""Harness task lifecycle control — cancel, resume, autonomy (FLOW-CTL, REL-ADV.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.applications._shared.task_control_governance import (
    TaskControlGovernanceBlockedError,
    build_cancel_task_execution_mutation_request,
    build_set_task_autonomy_mutation_request,
    enforce_task_control_authorization_result,
    is_task_execution_cancellable,
    task_execution_autonomy_revision,
    task_execution_state_revision,
    validate_task_control_principal_tenant_authority,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.autonomy_level import AutonomyLevel
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationAuthorizationEvidence,
    ControlPlaneMutationAuthorizationScope,
)
from intergrax.contracts.execution_identity import RunId, validate_run_id
from intergrax.contracts.human_approver import HumanApproverEvidence, local_development_approver_evidence
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.resume_planner import build_checkpoint_resume_task
from intergrax.runtime.task.active_task_registry import ActiveTaskBinding, ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from intergrax.runtime.task.task_contract import TaskPauseRecord
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner


class HitlResumeValidationError(ValueError):
    """Fail-closed validation for shared HITL resume surfaces."""


class TaskControlValidationError(ValueError):
    """Fail-closed validation for governed task-control surfaces."""


@dataclass(frozen=True, slots=True)
class TaskControlResult:
    task_id: str
    action: str
    accepted: bool
    detail: str = ""
    state: str | None = None
    metadata: dict[str, Any] | None = None
    authorization_evidence: ControlPlaneMutationAuthorizationEvidence | None = None
    authorization_scope: ControlPlaneMutationAuthorizationScope | None = None
    blocker_code: str | None = None
    policy_action: str | None = None


def _execute_cooperative_cancel(task: Task, *, reason: str) -> None:
    CancellationCoordinator.request(task, reason=reason)


def _blocked_result(
    *,
    task_id: str,
    action: str,
    detail: str,
    exc: TaskControlGovernanceBlockedError,
) -> TaskControlResult:
    return TaskControlResult(
        task_id=task_id,
        action=action,
        accepted=False,
        detail=detail,
        blocker_code=exc.blocker_code,
        policy_action=exc.policy_action,
        authorization_evidence=exc.authorization_evidence,
        authorization_scope=exc.authorization_scope,
        metadata={"tenant_scope_denial": exc.tenant_scope_denial.model_dump(mode="json")}
        if exc.tenant_scope_denial is not None
        else None,
    )


async def governed_cancel_active_task(
    *,
    task_id: str,
    run_id: str,
    mutation_id: str,
    principal: RequestIdentity,
    mutation_boundary: ControlPlaneMutationAuthorizationBoundary | None,
    reason: str = "operator_cancel",
    approval_evidence_ref: str | None = None,
) -> TaskControlResult:
    """Governed cooperative cancel for one exact active task/run binding."""
    normalized_mutation_id = mutation_id.strip()
    if not normalized_mutation_id:
        raise TaskControlValidationError("mutation_id_required")

    validated_run_id = validate_run_id(run_id)
    binding = await ActiveTaskRegistry.get(task_id)
    if binding is None:
        return TaskControlResult(
            task_id=task_id,
            action="cancel",
            accepted=False,
            detail="task_not_active",
        )

    if binding.run_id != validated_run_id:
        return TaskControlResult(
            task_id=task_id,
            action="cancel",
            accepted=False,
            detail="run_id_mismatch",
        )

    task = binding.task
    try:
        validate_task_control_principal_tenant_authority(
            principal=principal,
            task_tenant_id=task.tenant_id,
            task_id=binding.task_id,
            run_id=binding.run_id,
            operation="cancel_task_execution",
        )
    except TaskControlGovernanceBlockedError as exc:
        return _blocked_result(
            task_id=task_id,
            action="cancel",
            detail="tenant_authority_mismatch",
            exc=exc,
        )

    if not is_task_execution_cancellable(
        state=task.state,
        cancellation_requested=CancellationCoordinator.is_requested(task.metadata),
    ):
        return TaskControlResult(
            task_id=task_id,
            action="cancel",
            accepted=False,
            detail="task_not_cancellable",
            state=task.state.value,
        )

    if mutation_boundary is None:
        raise TaskControlGovernanceBlockedError(
            "TASK_CONTROL_BLOCKED_BY_MISSING_BOUNDARY",
            "cancel_task_execution requires ControlPlaneMutationAuthorizationBoundary",
            policy_action="DENY",
        )

    mutation_request = build_cancel_task_execution_mutation_request(
        principal=principal,
        tenant_id=task.tenant_id,
        task_id=binding.task_id,
        run_id=binding.run_id,
        mutation_id=normalized_mutation_id,
        current_state=task.state,
        approval_evidence_ref=approval_evidence_ref,
    )
    authorization_result = mutation_boundary.authorize(mutation_request)
    try:
        authorization_result = enforce_task_control_authorization_result(
            authorization_result,
            operation="cancel_task_execution",
        )
    except TaskControlGovernanceBlockedError as exc:
        return _blocked_result(
            task_id=task_id,
            action="cancel",
            detail=exc.blocker_code.lower(),
            exc=exc,
        )

    revalidated = await _revalidate_cancel_binding(
        task_id=task_id,
        expected_run_id=validated_run_id,
        expected_tenant_id=task.tenant_id,
        expected_current_revision=mutation_request.current_revision,
    )
    if revalidated is None:
        return TaskControlResult(
            task_id=task_id,
            action="cancel",
            accepted=False,
            detail="stale_active_binding",
            authorization_evidence=authorization_result.evidence,
        )

    _execute_cooperative_cancel(revalidated.task, reason=reason)
    return TaskControlResult(
        task_id=task_id,
        action="cancel",
        accepted=True,
        detail=reason,
        state=revalidated.task.state.value,
        authorization_evidence=authorization_result.evidence,
    )


async def _revalidate_cancel_binding(
    *,
    task_id: str,
    expected_run_id: RunId,
    expected_tenant_id: str,
    expected_current_revision: str,
) -> ActiveTaskBinding | None:
    binding = await ActiveTaskRegistry.get(task_id)
    if binding is None:
        return None
    if binding.run_id != expected_run_id:
        return None
    if binding.task.tenant_id != expected_tenant_id:
        return None
    if task_execution_state_revision(state=binding.task.state.value) != expected_current_revision:
        return None
    if not is_task_execution_cancellable(
        state=binding.task.state,
        cancellation_requested=CancellationCoordinator.is_requested(binding.task.metadata),
    ):
        return None
    return binding


def _execute_autonomy_change(task: Task, *, target_level: AutonomyLevel) -> AutonomyLevel | None:
    previous = task.options.governance.autonomy_level
    task.options.governance.autonomy_level = target_level
    task.metadata["autonomy_level"] = target_level.value
    task.metadata["autonomy_level_previous"] = previous.value if previous else None
    task.metadata["autonomy_level_changed"] = True
    task.sync_metadata()
    return previous


async def governed_set_task_autonomy(
    *,
    task_id: str,
    run_id: str,
    mutation_id: str,
    target_autonomy_level: AutonomyLevel,
    principal: RequestIdentity,
    mutation_boundary: ControlPlaneMutationAuthorizationBoundary | None,
    approval_evidence_ref: str | None = None,
) -> TaskControlResult:
    """Governed autonomy mutation for one exact active task/run binding."""
    normalized_mutation_id = mutation_id.strip()
    if not normalized_mutation_id:
        raise TaskControlValidationError("mutation_id_required")

    validated_run_id = validate_run_id(run_id)
    binding = await ActiveTaskRegistry.get(task_id)
    if binding is None:
        return TaskControlResult(
            task_id=task_id,
            action="set_autonomy",
            accepted=False,
            detail="task_not_active",
        )

    if binding.run_id != validated_run_id:
        return TaskControlResult(
            task_id=task_id,
            action="set_autonomy",
            accepted=False,
            detail="run_id_mismatch",
        )

    task = binding.task
    try:
        validate_task_control_principal_tenant_authority(
            principal=principal,
            task_tenant_id=task.tenant_id,
            task_id=binding.task_id,
            run_id=binding.run_id,
            operation="set_task_autonomy",
        )
    except TaskControlGovernanceBlockedError as exc:
        return _blocked_result(
            task_id=task_id,
            action="set_autonomy",
            detail="tenant_authority_mismatch",
            exc=exc,
        )

    current_autonomy = task.options.governance.autonomy_level
    if current_autonomy == target_autonomy_level:
        return TaskControlResult(
            task_id=task_id,
            action="set_autonomy",
            accepted=True,
            detail="already_at_target",
            state=task.state.value,
            metadata={"previous": current_autonomy.value if current_autonomy else None},
        )

    if mutation_boundary is None:
        raise TaskControlGovernanceBlockedError(
            "TASK_CONTROL_BLOCKED_BY_MISSING_BOUNDARY",
            "set_task_autonomy requires ControlPlaneMutationAuthorizationBoundary",
            policy_action="DENY",
        )

    mutation_request = build_set_task_autonomy_mutation_request(
        principal=principal,
        tenant_id=task.tenant_id,
        task_id=binding.task_id,
        run_id=binding.run_id,
        mutation_id=normalized_mutation_id,
        current_autonomy_level=current_autonomy,
        target_autonomy_level=target_autonomy_level,
        approval_evidence_ref=approval_evidence_ref,
    )
    authorization_result = mutation_boundary.authorize(mutation_request)
    try:
        authorization_result = enforce_task_control_authorization_result(
            authorization_result,
            operation="set_task_autonomy",
        )
    except TaskControlGovernanceBlockedError as exc:
        return _blocked_result(
            task_id=task_id,
            action="set_autonomy",
            detail=exc.blocker_code.lower(),
            exc=exc,
        )

    revalidated = await _revalidate_autonomy_binding(
        task_id=task_id,
        expected_run_id=validated_run_id,
        expected_tenant_id=task.tenant_id,
        expected_current_revision=mutation_request.current_revision,
    )
    if revalidated is None:
        return TaskControlResult(
            task_id=task_id,
            action="set_autonomy",
            accepted=False,
            detail="stale_active_binding",
            authorization_evidence=authorization_result.evidence,
        )

    previous = _execute_autonomy_change(revalidated.task, target_level=target_autonomy_level)
    return TaskControlResult(
        task_id=task_id,
        action="set_autonomy",
        accepted=True,
        detail=target_autonomy_level.value,
        state=revalidated.task.state.value,
        metadata={"previous": previous.value if previous else None},
        authorization_evidence=authorization_result.evidence,
    )


async def _revalidate_autonomy_binding(
    *,
    task_id: str,
    expected_run_id: RunId,
    expected_tenant_id: str,
    expected_current_revision: str,
) -> ActiveTaskBinding | None:
    binding = await ActiveTaskRegistry.get(task_id)
    if binding is None:
        return None
    if binding.run_id != expected_run_id:
        return None
    if binding.task.tenant_id != expected_tenant_id:
        return None
    current_revision = task_execution_autonomy_revision(
        autonomy_level=binding.task.options.governance.autonomy_level,
    )
    if current_revision != expected_current_revision:
        return None
    return binding


async def set_task_autonomy(task_id: str, level: AutonomyLevel) -> TaskControlResult:
    """Deprecated raw mutation — use ``governed_set_task_autonomy`` for supported surfaces."""
    binding = await ActiveTaskRegistry.get(task_id)
    if binding is None:
        return TaskControlResult(
            task_id=task_id,
            action="set_autonomy",
            accepted=False,
            detail="task_not_active",
        )
    task = binding.task
    previous = _execute_autonomy_change(task, target_level=level)
    return TaskControlResult(
        task_id=task_id,
        action="set_autonomy",
        accepted=True,
        detail=level.value,
        state=task.state.value,
        metadata={"previous": previous.value if previous else None},
    )


def _pause_record_from_checkpoint(checkpoint: TaskCheckpoint) -> TaskPauseRecord | None:
    snapshot = Task.model_validate(checkpoint.task_snapshot)
    return snapshot.runtime.governance.pause_record


def _materialize_hitl_resume_input(
    task: Task,
    *,
    checkpoint: TaskCheckpoint,
    operator_input: dict[str, Any] | None,
    approver: HumanApproverEvidence | None,
) -> None:
    verdict = (operator_input or {}).get("verdict")
    if not verdict:
        return

    pause_record = _pause_record_from_checkpoint(checkpoint)
    if pause_record is None:
        raise HitlResumeValidationError(
            "checkpoint has no active pause_record for human approval resume"
        )

    forged_pause_id = (operator_input or {}).get("pause_id")
    if forged_pause_id is not None and forged_pause_id != pause_record.pause_id:
        raise HitlResumeValidationError(
            "operator_input pause_id conflicts with checkpoint pause_record"
        )

    forged_request_id = (operator_input or {}).get("human_request_id")
    if forged_request_id is not None and forged_request_id != pause_record.human_request_id:
        raise HitlResumeValidationError(
            "operator_input human_request_id conflicts with checkpoint pause_record"
        )

    task.options.human.pause_id = pause_record.pause_id
    task.options.human.human_request_id = pause_record.human_request_id

    if approver is not None:
        task.options.human.approver = approver
    else:
        task.options.human.approver = local_development_approver_evidence(
            tenant_id=task.tenant_id,
        )


async def resume_task_with_token(
    runner: UnifiedTaskRunner,
    *,
    task_id: str,
    resume_token: str,
    operator_input: dict[str, Any] | None = None,
    checkpoint: TaskCheckpoint,
    approver: HumanApproverEvidence | None = None,
) -> TaskResult:
    task = build_checkpoint_resume_task(checkpoint)
    task.task_id = task_id
    task.options.long_running.resume_token = resume_token
    if operator_input:
        verdict = operator_input.get("verdict")
        if verdict:
            task.options.human.verdict = str(verdict)
        response_text = operator_input.get("response_text")
        if response_text:
            task.options.human.response_text = str(response_text)
    _materialize_hitl_resume_input(
        task,
        checkpoint=checkpoint,
        operator_input=operator_input,
        approver=approver,
    )
    return await runner.run_task(task, resume_checkpoint=checkpoint)
