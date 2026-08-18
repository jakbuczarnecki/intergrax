# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.runtime.human.escalation import EscalationTarget
from intergrax.runtime.human.models import HumanResponseVerdict, build_human_decision_record
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.human.persistence_contract import HumanDecisionPersistence
from intergrax.runtime.task.task import Task


def normalize_human_response(task: Task) -> None:
    response = task.options.human.response_text
    if response and task.options.human.verdict is None:
        HumanPauseCoordinator.record_human_response(task, str(response))


def clear_consumed_human_input(task: Task) -> None:
    task.options.human.response_text = None
    task.options.human.verdict = None
    task.options.human.pause_id = None
    task.options.human.human_request_id = None
    task.sync_metadata()


def persist_human_decision(
    task: Task,
    verdict: HumanResponseVerdict,
    *,
    human_store: HumanDecisionPersistence | None,
    response_text: str = "",
) -> None:
    if human_store is None:
        return
    human_request = HumanPauseCoordinator.human_request_from_task(task)
    target_raw = task.runtime.governance.escalation_target
    target = EscalationTarget(str(target_raw)) if target_raw else None
    record = build_human_decision_record(
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
