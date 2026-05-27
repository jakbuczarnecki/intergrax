# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Human-in-the-loop pause and resume helpers (architecture §42.9, §42.10)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from intergrax.contracts.agent_decision import HumanRequest
from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.execution_interrupt import ExecutionInterrupt
from intergrax.runtime.interrupts.handler import GovernanceResolution
from intergrax.runtime.task.task import Task

GOVERNANCE_HUMAN_REQUEST_KEY = "governance_human_request"
GOVERNANCE_INTERRUPT_KEY = "governance_interrupt"
GOVERNANCE_PAUSE_KEY = "governance_pause"
HUMAN_APPROVED_KEY = "human_approved"
HUMAN_RESPONSE_KEY = "human_response"


class PauseRecord(BaseModel):
    pause_id: str = Field(default_factory=lambda: f"pause_{uuid4().hex[:12]}")
    task_id: str
    human_request_id: str
    reason: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = "pause_record.v1"


class HumanPauseCoordinator:
    """Persists pause metadata on tasks and supports resume signals."""

    @staticmethod
    def apply_pause(task: Task, execution: AgentExecutionResult) -> Task:
        if execution.human_request is not None:
            task.metadata[GOVERNANCE_HUMAN_REQUEST_KEY] = execution.human_request.model_dump()
        if execution.execution_interrupt is not None:
            task.metadata[GOVERNANCE_INTERRUPT_KEY] = execution.execution_interrupt.model_dump()
        if execution.human_request is not None:
            reason = ""
            if execution.agent_decision is not None:
                reason = execution.agent_decision.reason
            task.metadata["governance_pause_record"] = PauseRecord(
                task_id=task.task_id,
                human_request_id=execution.human_request.request_id,
                reason=reason,
            ).model_dump()
        task.metadata[GOVERNANCE_PAUSE_KEY] = True
        return task

    @staticmethod
    def apply_resolution(task: Task, resolution: GovernanceResolution) -> Task:
        if resolution.human_request is not None:
            task.metadata[GOVERNANCE_HUMAN_REQUEST_KEY] = resolution.human_request.model_dump()
        if resolution.interrupt is not None:
            task.metadata[GOVERNANCE_INTERRUPT_KEY] = resolution.interrupt.model_dump()
        task.metadata[GOVERNANCE_PAUSE_KEY] = True
        return task

    @staticmethod
    def clear_pause(task: Task) -> Task:
        task.metadata.pop(GOVERNANCE_PAUSE_KEY, None)
        return task

    @staticmethod
    def is_resumed(task: Task) -> bool:
        return bool(task.metadata.get(HUMAN_APPROVED_KEY))

    @staticmethod
    def record_human_response(task: Task, response: str) -> Task:
        normalized = response.strip().lower()
        task.metadata[HUMAN_RESPONSE_KEY] = response
        task.metadata[HUMAN_APPROVED_KEY] = normalized in {"approve", "approved", "yes", "accept"}
        return task

    @staticmethod
    def human_request_from_task(task: Task) -> Optional[HumanRequest]:
        raw = task.metadata.get(GOVERNANCE_HUMAN_REQUEST_KEY)
        if not raw:
            return None
        return HumanRequest.model_validate(raw)

    @staticmethod
    def interrupt_from_task(task: Task) -> Optional[ExecutionInterrupt]:
        raw = task.metadata.get(GOVERNANCE_INTERRUPT_KEY)
        if not raw:
            return None
        return ExecutionInterrupt.model_validate(raw)
