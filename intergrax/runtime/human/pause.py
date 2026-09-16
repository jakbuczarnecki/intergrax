# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Human-in-the-loop pause projection, evidence, and compatibility helpers (§42.9, §42.38).

GR-5-R3: canonical pause/resume lifecycle authority is
:class:`intergrax.contracts.execution_continuation.ExecutionContinuationPort`.
``HumanPauseCoordinator`` projects canonical continuation onto Task/Human state and
records human evidence — it does **not** own lifecycle truth.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from intergrax.contracts.agent_decision import HumanRequest
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationLifecycleState,
    ExecutionContinuationLookup,
    ExecutionContinuationPort,
    ExecutionContinuationResolutionCommand,
    ExecutionHumanVerdict,
    PendingExecutionContinuation,
    execution_continuation_resolution_command_for_pending_human_verdict,
)
from intergrax.contracts.execution_continuation_projection import (
    ExecutionContinuationCanonicalProjectionApplyError,
    ExecutionContinuationProjectionError,
    ExecutionContinuationProjectionResult,
    ExecutionContinuationProjectionSink,
    ExecutionContinuationProjectionStatus,
)
from intergrax.contracts.execution_identity import (
    peek_active_execution_id,
    peek_active_execution_identity,
    require_active_execution_id,
    require_active_execution_identity,
    validate_attempt_id,
    validate_execution_id,
)
from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.declarative_hitl import DeclarativeHitlPendingApproval
from intergrax.contracts.execution_interrupt import ExecutionInterrupt
from intergrax.runtime.human.response_parser import parse_human_response
from intergrax.runtime.human.request_contract import HumanTimeoutCoordinator
from intergrax.contracts.human_approver import HumanApproverEvidence
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.interrupts.handler import GovernanceResolution
from intergrax.runtime.task.execution_continuation_projection import (
    apply_canonical_projection_fields,
    wire_task_execution_continuation_projection_sink,
)
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_contract import (
    HumanApprovalResolution,
    TaskPauseRecord,
)
from intergrax.runtime.task.task_metadata_keys import (
    ESCALATION_CHAIN_KEY,
    ESCALATION_LEVEL_KEY,
    ESCALATION_TARGET_KEY,
    GOVERNANCE_HUMAN_REQUEST_KEY,
    GOVERNANCE_INTERRUPT_KEY,
    GOVERNANCE_PAUSE_KEY,
    HUMAN_APPROVED_KEY,
    HUMAN_DECISION_KEY,
    HUMAN_ESCALATED_KEY,
    HUMAN_REJECTED_KEY,
    HUMAN_RESPONSE_KEY,
    TaskMetadataKey,
)

__all__ = [
    "ESCALATION_CHAIN_KEY",
    "ESCALATION_LEVEL_KEY",
    "ESCALATION_TARGET_KEY",
    "GOVERNANCE_HUMAN_REQUEST_KEY",
    "GOVERNANCE_INTERRUPT_KEY",
    "GOVERNANCE_PAUSE_KEY",
    "HUMAN_APPROVED_KEY",
    "HUMAN_DECISION_KEY",
    "HUMAN_ESCALATED_KEY",
    "HUMAN_REJECTED_KEY",
    "HUMAN_RESPONSE_KEY",
    "HumanApprovalResolutionError",
    "HumanPauseCoordinator",
    "PauseRecord",
    "TaskMetadataKey",
    "approved_resolution_for_resume",
]


class HumanApprovalResolutionError(ValueError):
    """Fail-closed human approval resolution against the active pause/request."""


class PauseRecord(BaseModel):
    pause_id: str = Field(default_factory=lambda: f"pause_{uuid4().hex[:12]}")
    task_id: str
    human_request_id: str
    reason: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = "pause_record.v1"


def approved_resolution_for_resume(
    *,
    task_id: str,
    resolution: HumanApprovalResolution | None,
    expected_pause_id: str,
    expected_human_request_id: str,
    run_id: str | None = None,
) -> HumanApprovalResolution | None:
    """Return canonical APPROVE resolution only when it matches the exact lifecycle."""
    if resolution is None:
        return None
    if resolution.verdict is not HumanResponseVerdict.APPROVE:
        return None
    if resolution.task_id != task_id:
        return None
    if resolution.pause_id != expected_pause_id:
        return None
    if resolution.human_request_id != expected_human_request_id:
        return None
    if run_id is not None and resolution.run_id is not None and resolution.run_id != run_id:
        return None
    return resolution


class HumanPauseCoordinator:
    """Projects canonical continuation onto Task/Human representation (compatibility facade).

    Legacy graph paths may still call :meth:`apply_pause` without a canonical snapshot;
    that path is non-authoritative and mints local pause identifiers. Canonical GR-5 flows
    must use :meth:`project_continuation` after ``ExecutionContinuationPort`` transitions.
    """

    @staticmethod
    def approved_resolution_for_resume(
        *,
        task_id: str,
        resolution: HumanApprovalResolution | None,
        expected_pause_id: str,
        expected_human_request_id: str,
        run_id: str | None = None,
    ) -> HumanApprovalResolution | None:
        return approved_resolution_for_resume(
            task_id=task_id,
            resolution=resolution,
            expected_pause_id=expected_pause_id,
            expected_human_request_id=expected_human_request_id,
            run_id=run_id,
        )

    @staticmethod
    def project_continuation(
        task: Task,
        pending: PendingExecutionContinuation,
        projection_sink: ExecutionContinuationProjectionSink | None = None,
        *,
        accepted_hitl_resolution: HumanApprovalResolution | None = None,
    ) -> ExecutionContinuationProjectionResult:
        """Project one canonical continuation snapshot onto Task (idempotent per revision)."""
        if projection_sink is not None:
            return projection_sink.project(pending)
        if accepted_hitl_resolution is not None:
            try:
                apply_canonical_projection_fields(
                    task,
                    pending,
                    accepted_hitl_resolution=accepted_hitl_resolution,
                )
            except ExecutionContinuationProjectionError:
                raise
            except Exception as exc:
                raise ExecutionContinuationProjectionError(str(exc)) from exc
            return ExecutionContinuationProjectionResult(
                status=ExecutionContinuationProjectionStatus.APPLIED,
                applied_revision=pending.revision,
            )
        return wire_task_execution_continuation_projection_sink(task).project(pending)

    @staticmethod
    def apply_pause(
        task: Task,
        execution: AgentExecutionResult,
        *,
        pause_id: str | None = None,
    ) -> Task:
        """Legacy/projection helper — does **not** establish canonical PAUSED/WAITING."""
        gov = task.runtime.governance
        if execution.human_request is not None:
            gov.hitl_resolution = None
            gov.governed_continuation_grant = None
            gov.physical_delegation_continuation_grant = None
            HumanTimeoutCoordinator.attach_to_task(task, execution.human_request)
        if execution.execution_interrupt is not None:
            gov.execution_interrupt = execution.execution_interrupt
        if execution.human_request is not None:
            reason = ""
            if execution.agent_decision is not None:
                reason = execution.agent_decision.reason
            resolved_pause_id = pause_id if pause_id is not None else f"pause_{uuid4().hex[:12]}"
            record = PauseRecord(
                pause_id=resolved_pause_id,
                task_id=task.task_id,
                human_request_id=execution.human_request.request_id,
                reason=reason,
            )
            gov.pause_record = TaskPauseRecord(
                pause_id=record.pause_id,
                task_id=record.task_id,
                human_request_id=record.human_request_id,
                reason=record.reason,
                created_at=record.created_at.isoformat(),
                schema_version=record.schema_version,
            )
        if execution.declarative_hitl_pending is not None:
            pending = execution.declarative_hitl_pending
            if gov.pause_record is not None:
                pending = pending.model_copy(
                    update={
                        "human_request_id": gov.pause_record.human_request_id,
                        "pause_id": gov.pause_record.pause_id,
                    }
                )
            gov.declarative_hitl_pending = pending
        gov.paused = True
        task.sync_metadata()
        return task

    @staticmethod
    def apply_resolution(task: Task, resolution: GovernanceResolution) -> Task:
        """Project governance interrupt metadata — does **not** authorize canonical resume."""
        gov = task.runtime.governance
        if resolution.human_request is not None:
            HumanTimeoutCoordinator.attach_to_task(task, resolution.human_request)
        if resolution.interrupt is not None:
            gov.execution_interrupt = resolution.interrupt
        gov.paused = True
        task.sync_metadata()
        return task

    @staticmethod
    def clear_pause(task: Task) -> Task:
        """Clear Task pause projection only — does **not** establish canonical RESUMED."""
        task.runtime.governance.paused = False
        task.sync_metadata()
        return task

    @staticmethod
    def _validate_human_response_inputs(
        task: Task,
        verdict: HumanResponseVerdict,
        *,
        approver: HumanApproverEvidence,
        pause_id: str | None,
        human_request_id: str | None,
        attempt_id: str | None,
        execution_id: str | None,
        require_unresolved_hitl: bool,
    ) -> tuple[str, str, str | None, str | None]:
        gov = task.runtime.governance
        if require_unresolved_hitl and gov.hitl_resolution is not None:
            raise HumanApprovalResolutionError("human approval already resolved")

        pause_record = gov.pause_record
        if pause_record is None:
            raise HumanApprovalResolutionError("no active pause record")

        if pause_record.task_id != task.task_id:
            raise HumanApprovalResolutionError("pause task_id mismatch")

        if verdict is HumanResponseVerdict.UNKNOWN:
            raise HumanApprovalResolutionError("unsupported verdict")

        if approver.tenant_id != task.tenant_id:
            raise HumanApprovalResolutionError("approver tenant_id mismatch")

        if pause_id is None:
            raise HumanApprovalResolutionError("pause_id required")

        if human_request_id is None:
            raise HumanApprovalResolutionError("human_request_id required")

        active_pause_id = pause_record.pause_id
        active_request_id = pause_record.human_request_id

        if pause_id != active_pause_id:
            raise HumanApprovalResolutionError("pause_id mismatch")

        if human_request_id != active_request_id:
            raise HumanApprovalResolutionError("human_request_id mismatch")

        if gov.human_request is not None:
            if gov.human_request.request_id != active_request_id:
                raise HumanApprovalResolutionError("human_request identity mismatch")

        governed = (
            gov.human_request.governed_continuation
            if gov.human_request is not None
            else None
        )
        resolved_attempt: str | None = None
        resolved_execution: str | None = None
        if governed is not None:
            has_attempt = attempt_id is not None
            has_execution = execution_id is not None
            if has_attempt != has_execution:
                raise HumanApprovalResolutionError(
                    "attempt_id and execution_id must be supplied together for governed continuation",
                )
            if has_attempt:
                resolved_attempt = str(validate_attempt_id(attempt_id))
                resolved_execution = str(validate_execution_id(execution_id))
                active_identity = peek_active_execution_identity()
                if active_identity is not None:
                    _, active_attempt = active_identity
                    active_execution = peek_active_execution_id()
                    if active_execution is None:
                        raise HumanApprovalResolutionError("active ExecutionId required")
                    if (
                        str(active_attempt) != resolved_attempt
                        or str(active_execution) != resolved_execution
                    ):
                        raise HumanApprovalResolutionError(
                            "execution identity does not match active execution",
                        )
            else:
                _, active_attempt = require_active_execution_identity()
                active_execution = require_active_execution_id()
                resolved_attempt = str(active_attempt)
                resolved_execution = str(active_execution)
            if str(governed.attempt_id) != resolved_attempt:
                raise HumanApprovalResolutionError("governed continuation attempt_id mismatch")
            if str(governed.execution_id) != resolved_execution:
                raise HumanApprovalResolutionError(
                    "governed continuation execution_id mismatch",
                )

        return active_pause_id, active_request_id, resolved_attempt, resolved_execution

    @staticmethod
    def _validate_human_response_against_pending(
        pending: PendingExecutionContinuation,
        *,
        pause_id: str,
        human_request_id: str,
    ) -> None:
        if pending.pause_id is None:
            raise HumanApprovalResolutionError("canonical pending pause_id required")
        if pending.human_request_id is None:
            raise HumanApprovalResolutionError("canonical pending human_request_id required")
        if pause_id != pending.pause_id:
            raise HumanApprovalResolutionError("pause_id mismatch")
        if human_request_id != pending.human_request_id:
            raise HumanApprovalResolutionError("human_request_id mismatch")

    @staticmethod
    def _build_human_approval_resolution(
        task: Task,
        verdict: HumanResponseVerdict,
        *,
        approver: HumanApproverEvidence,
        pause_id: str,
        human_request_id: str,
        run_id: str | None,
        attempt_id: str | None,
        execution_id: str | None,
        response_text: str | None,
    ) -> HumanApprovalResolution:
        return HumanApprovalResolution(
            task_id=task.task_id,
            pause_id=pause_id,
            human_request_id=human_request_id,
            verdict=verdict,
            approver=approver,
            resolved_at=datetime.now(timezone.utc).isoformat(),
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            response_text=response_text or task.options.human.response_text,
        )

    @staticmethod
    def resolve_human_response(
        task: Task,
        verdict: HumanResponseVerdict,
        *,
        approver: HumanApproverEvidence,
        pause_id: str | None = None,
        human_request_id: str | None = None,
        run_id: str | None = None,
        attempt_id: str | None = None,
        execution_id: str | None = None,
        response_text: str | None = None,
    ) -> HumanApprovalResolution:
        """Record typed human decision evidence on Task — not canonical lifecycle authority."""
        active_pause_id, active_request_id, resolved_attempt, resolved_execution = (
            HumanPauseCoordinator._validate_human_response_inputs(
                task,
                verdict,
                approver=approver,
                pause_id=pause_id,
                human_request_id=human_request_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                require_unresolved_hitl=True,
            )
        )
        resolution = HumanPauseCoordinator._build_human_approval_resolution(
            task,
            verdict,
            approver=approver,
            pause_id=active_pause_id,
            human_request_id=active_request_id,
            run_id=run_id,
            attempt_id=resolved_attempt,
            execution_id=resolved_execution,
            response_text=response_text,
        )
        gov = task.runtime.governance
        gov.hitl_resolution = resolution
        task.sync_metadata()
        return resolution

    @staticmethod
    def resolve_human_response_and_apply_canonical(
        task: Task,
        verdict: HumanResponseVerdict,
        *,
        approver: HumanApproverEvidence,
        continuation: ExecutionContinuationPort,
        projection_sink: ExecutionContinuationProjectionSink | None = None,
        pause_id: str | None = None,
        human_request_id: str | None = None,
        run_id: str | None = None,
        attempt_id: str | None = None,
        execution_id: str | None = None,
        response_text: str | None = None,
    ) -> PendingExecutionContinuation:
        """Validate input, canonical ``apply_resolution``, then Task projection."""
        gov = task.runtime.governance
        human_request = gov.human_request
        if human_request is None:
            raise HumanApprovalResolutionError("human_request required for canonical resolution")
        governed = human_request.governed_continuation
        continuation_id = (
            governed.continuation_request_id
            if governed is not None
            else f"gcr_hr_{human_request.request_id}"
        )
        pending = continuation.get_pending(
            ExecutionContinuationLookup(continuation_id=continuation_id),
        )
        if pending.governed_correlation is not None:
            gov.human_request = human_request.model_copy(
                update={"governed_continuation": pending.governed_correlation},
            )
        elif human_request.governed_continuation is None:
            raise HumanApprovalResolutionError(
                "governed continuation correlation required for canonical resolution",
            )
        active_pause_id, active_request_id, resolved_attempt, resolved_execution = (
            HumanPauseCoordinator._validate_human_response_inputs(
                task,
                verdict,
                approver=approver,
                pause_id=pause_id,
                human_request_id=human_request_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                require_unresolved_hitl=True,
            )
        )
        governed = (
            gov.human_request.governed_continuation if gov.human_request is not None else None
        )
        if governed is None:
            governed = pending.governed_correlation
        if governed is None:
            raise HumanApprovalResolutionError(
                "governed continuation correlation required for canonical resolution",
            )
        HumanPauseCoordinator._validate_human_response_against_pending(
            pending,
            pause_id=active_pause_id,
            human_request_id=active_request_id,
        )
        correlation = pending.governed_correlation or governed
        if verdict is HumanResponseVerdict.APPROVE:
            execution_verdict = ExecutionHumanVerdict.APPROVE
        elif verdict is HumanResponseVerdict.REJECT:
            execution_verdict = ExecutionHumanVerdict.REJECT
        elif verdict is HumanResponseVerdict.ESCALATE:
            execution_verdict = ExecutionHumanVerdict.ESCALATE
        else:
            raise HumanApprovalResolutionError("unsupported verdict for canonical resolution")
        if pending.pause_id is None or pending.human_request_id is None:
            raise HumanApprovalResolutionError(
                "canonical pending pause_id and human_request_id required",
            )
        command = execution_continuation_resolution_command_for_pending_human_verdict(
            pending,
            verdict=execution_verdict,
            approver=approver,
            human_request_id=pending.human_request_id,
            resolved_at=datetime.now(timezone.utc).isoformat(),
        )
        updated = continuation.apply_resolution(command)
        accepted_resolution = HumanPauseCoordinator._build_human_approval_resolution(
            task,
            verdict,
            approver=approver,
            pause_id=pending.pause_id,
            human_request_id=pending.human_request_id,
            run_id=run_id,
            attempt_id=resolved_attempt,
            execution_id=resolved_execution,
            response_text=response_text,
        )
        try:
            HumanPauseCoordinator.project_continuation(
                task,
                updated,
                projection_sink=projection_sink,
                accepted_hitl_resolution=accepted_resolution,
            )
        except ExecutionContinuationProjectionError as exc:
            raise ExecutionContinuationCanonicalProjectionApplyError(
                str(exc),
                canonical_snapshot=updated,
            ) from exc
        return updated

    @staticmethod
    def verdict_from_task(task: Task) -> Optional[HumanResponseVerdict]:
        raw = task.options.human.verdict
        if not raw:
            return None
        try:
            return HumanResponseVerdict(str(raw))
        except ValueError:
            return HumanResponseVerdict.UNKNOWN

    @staticmethod
    def is_resumed(task: Task) -> bool:
        projected = task.runtime.governance.projected_continuation_lifecycle_state
        if projected is not None:
            return projected == ExecutionContinuationLifecycleState.RESUMED.value
        return task.options.human.is_resumed

    @staticmethod
    def is_rejected(task: Task) -> bool:
        projected = task.runtime.governance.projected_continuation_lifecycle_state
        if projected is not None:
            return projected == ExecutionContinuationLifecycleState.REJECTED.value
        return task.options.human.is_rejected

    @staticmethod
    def is_escalated(task: Task) -> bool:
        projected = task.runtime.governance.projected_continuation_lifecycle_state
        if projected is not None:
            return projected == ExecutionContinuationLifecycleState.ESCALATED.value
        return task.options.human.is_escalated

    @staticmethod
    def record_human_response(task: Task, response: str) -> Task:
        verdict = parse_human_response(response)
        task.options.human.response_text = response
        task.options.human.verdict = verdict.value
        task.sync_metadata()
        return task

    @staticmethod
    def human_request_from_task(task: Task) -> Optional[HumanRequest]:
        return task.runtime.governance.human_request

    @staticmethod
    def interrupt_from_task(task: Task) -> Optional[ExecutionInterrupt]:
        return task.runtime.governance.execution_interrupt

    @staticmethod
    def escalation_level(task: Task) -> int:
        return task.runtime.governance.escalation_level

    @staticmethod
    def escalation_chain(task: Task) -> list:
        return [step.model_dump() for step in task.runtime.governance.escalation_chain]
