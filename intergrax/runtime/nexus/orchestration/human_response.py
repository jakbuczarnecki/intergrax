# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.runtime.human.escalation import EscalationTarget
from intergrax.runtime.human.models import HumanResponseVerdict, build_human_decision_record
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.human.persistence_contract import HumanDecisionPersistence
from intergrax.runtime.task.task import Task


class HumanDecisionPersistenceError(ValueError):
    """Raised when human decision evidence is incomplete for persistence."""


def normalize_human_response(task: Task) -> None:
    response = task.options.human.response_text
    if response and task.options.human.verdict is None:
        HumanPauseCoordinator.record_human_response(task, str(response))


def clear_consumed_human_input(task: Task) -> None:
    task.options.human.response_text = None
    task.options.human.verdict = None
    task.options.human.pause_id = None
    task.options.human.human_request_id = None
    task.options.human.approver = None
    task.sync_metadata()


def _approver_from_resolution(task: Task) -> object:
    resolution = task.runtime.governance.hitl_resolution
    if resolution is None:
        raise HumanDecisionPersistenceError(
            "cannot persist human decision without canonical approval resolution"
        )
    return resolution.approver


def _canonical_run_id_for_persistence(
    task: Task,
    *,
    run_id: str | None = None,
) -> str | None:
    if run_id is not None:
        return run_id
    resolution = task.runtime.governance.hitl_resolution
    if resolution is None:
        return None
    return resolution.run_id


def persist_human_decision(
    task: Task,
    verdict: HumanResponseVerdict,
    *,
    human_store: HumanDecisionPersistence | None,
    response_text: str = "",
    run_id: str | None = None,
) -> None:
    if human_store is None:
        return
    approver = _approver_from_resolution(task)
    human_request = HumanPauseCoordinator.human_request_from_task(task)
    target_raw = task.runtime.governance.escalation_target
    target = EscalationTarget(str(target_raw)) if target_raw else None
    canonical_run_id = _canonical_run_id_for_persistence(task, run_id=run_id)
    record = build_human_decision_record(
        task_id=task.task_id,
        tenant_id=task.tenant_id,
        approver=approver,  # type: ignore[arg-type]
        verdict=verdict,
        response_text=response_text or str(task.options.human.response_text or ""),
        human_request_id=human_request.request_id if human_request else "",
        escalation_level=HumanPauseCoordinator.escalation_level(task),
        escalation_target=target,
        agent_id=task.agent_id,
        run_id=canonical_run_id,
        task_subject_user_id=task.user_id,
    )
    human_store.record(record)
