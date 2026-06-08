# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Parse human responses and route escalations (Phase F.3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intergrax.runtime.human.models import (
    EscalationOutcome,
    EscalationTarget,
    HumanResponseVerdict,
)
from intergrax.runtime.human.response_parser import parse_human_response
from intergrax.runtime.task.task_contract import EscalationStep
from intergrax.runtime.task.task_metadata_keys import TaskMetadataKey

if TYPE_CHECKING:
    from intergrax.runtime.task.task import Task

__all__ = [
    "EscalationRouter",
    "TaskMetadataKey",
    "parse_human_response",
]


class EscalationRouter:
    """Routes HITL escalations per §42.38 (minimal laboratory policy)."""

    def __init__(self, *, max_levels: int = 3) -> None:
        self._max_levels = max(1, max_levels)

    def route(self, task: Task) -> EscalationOutcome:
        level = task.runtime.governance.escalation_level + 1
        if level >= self._max_levels:
            return EscalationOutcome(
                target=EscalationTarget.FAIL_TASK,
                level=level,
                message="escalation limit reached",
                fail_task=True,
            )
        if level >= 2:
            return EscalationOutcome(
                target=EscalationTarget.APPLICATION_ADMIN,
                level=level,
                message="escalated to application admin",
            )
        return EscalationOutcome(
            target=EscalationTarget.HUMAN_OPERATOR,
            level=level,
            message="escalated to human operator",
        )

    def apply_to_task(self, task: Task, outcome: EscalationOutcome) -> Task:
        gov = task.runtime.governance
        gov.escalation_level = outcome.level
        gov.escalation_target = outcome.target.value
        gov.escalation_chain.append(
            EscalationStep(
                level=outcome.level,
                target=outcome.target.value,
                message=outcome.message,
            )
        )
        task.sync_metadata()
        return task
