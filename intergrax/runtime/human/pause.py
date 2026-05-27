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
from intergrax.contracts.execution_interrupt import ExecutionInterrupt
from intergrax.runtime.human.response_parser import parse_human_response
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.interrupts.handler import GovernanceResolution
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_contract import TaskPauseRecord
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
    "HumanPauseCoordinator",
    "PauseRecord",
]


class PauseRecord(BaseModel):
    pause_id: str = Field(default_factory=lambda: f"pause_{uuid4().hex[:12]}")
    task_id: str
    human_request_id: str
    reason: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = "pause_record.v1"


class HumanPauseCoordinator:
    """Persists pause state on tasks and supports approve/reject/escalate resume signals."""

    @staticmethod
    def apply_pause(task: Task, execution: AgentExecutionResult) -> Task:
        gov = task.runtime.governance
        if execution.human_request is not None:
            gov.human_request = execution.human_request
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
        gov.paused = True
        task.sync_metadata()
        return task

    @staticmethod
    def apply_resolution(task: Task, resolution: GovernanceResolution) -> Task:
        gov = task.runtime.governance
        if resolution.human_request is not None:
            gov.human_request = resolution.human_request
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
