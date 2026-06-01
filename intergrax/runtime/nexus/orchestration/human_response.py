# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.runtime.human.escalation import EscalationTarget
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.human.store import SQLiteHumanDecisionStore
from intergrax.runtime.task.task import Task


def normalize_human_response(task: Task) -> None:
    response = task.options.human.response_text
    if response and task.options.human.verdict is None:
        HumanPauseCoordinator.record_human_response(task, str(response))


def persist_human_decision(
    task: Task,
    verdict: HumanResponseVerdict,
    *,
    human_store: SQLiteHumanDecisionStore | None,
    response_text: str = "",
) -> None:
    if human_store is None:
        return
    human_request = HumanPauseCoordinator.human_request_from_task(task)
    target_raw = task.runtime.governance.escalation_target
    target = EscalationTarget(str(target_raw)) if target_raw else None
    record = SQLiteHumanDecisionStore.build_record(
        task_id=task.task_id,
        tenant_id=task.tenant_id,
        user_id=task.user_id,
        verdict=verdict,
        response_text=response_text or str(task.options.human.response_text or ""),
        human_request_id=human_request.request_id if human_request else "",
        escalation_level=HumanPauseCoordinator.escalation_level(task),
        escalation_target=target,
        agent_id=task.agent_id,
        run_id=task.task_id,
    )
    human_store.record(record)
