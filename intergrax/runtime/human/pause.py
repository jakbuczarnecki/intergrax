# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Human-in-the-loop pause, resume, reject and escalation helpers (§42.9, §42.38)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from intergrax.contracts.agent_decision import HumanRequest
from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.declarative_hitl import DeclarativeHitlPendingApproval
from intergrax.contracts.execution_interrupt import ExecutionInterrupt
from intergrax.runtime.human.response_parser import parse_human_response
from intergrax.runtime.human.request_contract import HumanTimeoutCoordinator
from intergrax.contracts.human_approver import HumanApproverEvidence
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.interrupts.handler import GovernanceResolution
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
    """Persists pause state on tasks and supports approve/reject/escalate resume signals."""

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
    def apply_pause(task: Task, execution: AgentExecutionResult) -> Task:
        gov = task.runtime.governance
        if execution.human_request is not None:
            gov.hitl_resolution = None
            gov.governed_continuation_grant = None
            HumanTimeoutCoordinator.attach_to_task(task, execution.human_request)
        if execution.execution_interrupt is not None:
            gov.execution_interrupt = execution.execution_interrupt
        if execution.human_request is not None:
            reason = ""
            if execution.agent_decision is not None:
                reason = execution.agent_decision.reason
            record = PauseRecord(
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
        task.runtime.governance.paused = False
        task.sync_metadata()
        return task

    @staticmethod
    def resolve_human_response(
        task: Task,
        verdict: HumanResponseVerdict,
        *,
        approver: HumanApproverEvidence,
        pause_id: str | None = None,
        human_request_id: str | None = None,
        run_id: str | None = None,
        response_text: str | None = None,
    ) -> HumanApprovalResolution:
        gov = task.runtime.governance
        if gov.hitl_resolution is not None:
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

        resolution = HumanApprovalResolution(
            task_id=task.task_id,
            pause_id=active_pause_id,
            human_request_id=active_request_id,
            verdict=verdict,
            approver=approver,
            resolved_at=datetime.now(timezone.utc).isoformat(),
            run_id=run_id,
            response_text=response_text or task.options.human.response_text,
        )
        gov.hitl_resolution = resolution
        task.sync_metadata()
        return resolution

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
        return task.options.human.is_resumed

    @staticmethod
    def is_rejected(task: Task) -> bool:
        return task.options.human.is_rejected

    @staticmethod
    def is_escalated(task: Task) -> bool:
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
