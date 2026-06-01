# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Retry facade for run-level and graph-level retries (Phase Q+-N.3, §31.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Optional

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.trace_bridge import runtime_event_from_task_state
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.retry.retry_types import RetryRecord
from intergrax.runtime.task.task import Task


@dataclass(frozen=True, slots=True)
class RetryCoordinator:
    """Unifies run retry policy checks and ``RETRY_SCHEDULED`` runtime events."""

    max_run_retries: int
    retry_run_on: FrozenSet[RuntimeErrorCode]

    def should_retry_run(
        self,
        *,
        attempt: int,
        error_code: RuntimeErrorCode,
    ) -> bool:
        return error_code in self.retry_run_on and attempt < self.max_run_retries

    @staticmethod
    def build_scheduled_event(
        task: Task,
        *,
        run_id: str,
        attempt: int,
        max_retries: int,
        reason: str,
        scope: str,
        alternate_agent_id: Optional[str] = None,
    ) -> RuntimeEvent:
        return runtime_event_from_task_state(
            task,
            run_id=run_id,
            message=f"retry scheduled ({scope}): {reason}",
        ).model_copy(
            update={
                "event_type": RuntimeEventType.RETRY_SCHEDULED,
                "phase": ExecutionPhase.RETRY_HANDLING,
                "payload": {
                    "scope": scope,
                    "attempt": attempt,
                    "max_retries": max_retries,
                    "reason": reason,
                    "alternate_agent_id": alternate_agent_id,
                },
            }
        )

    def scheduled_event_for_run_retry(
        self,
        task: Task,
        *,
        run_id: str,
        attempt: int,
        error_code: RuntimeErrorCode,
    ) -> RuntimeEvent:
        return self.build_scheduled_event(
            task,
            run_id=run_id,
            attempt=attempt,
            max_retries=self.max_run_retries,
            reason=error_code.value,
            scope="run",
        )

    def scheduled_event_for_agent_retry(
        self,
        task: Task,
        *,
        run_id: str,
        record: RetryRecord,
    ) -> RuntimeEvent:
        return self.build_scheduled_event(
            task,
            run_id=run_id,
            attempt=record.attempt,
            max_retries=0,
            reason=record.reason,
            scope="agent",
            alternate_agent_id=record.alternate_agent_id,
        )
