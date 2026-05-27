# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.human.escalation import EscalationRouter, parse_human_response
from intergrax.runtime.human.models import (
    EscalationTarget,
    HumanResponseVerdict,
)
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.human.store import SQLiteHumanDecisionStore
from intergrax.runtime.task.task import Task

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_parse_human_response_verdicts():
    assert parse_human_response("approve") == HumanResponseVerdict.APPROVE
    assert parse_human_response("reject") == HumanResponseVerdict.REJECT
    assert parse_human_response("escalate") == HumanResponseVerdict.ESCALATE
    assert parse_human_response("maybe") == HumanResponseVerdict.UNKNOWN


def test_human_pause_coordinator_reject_and_escalate():
    task = Task(tenant_id="t1", user_id="u1", message="x")
    HumanPauseCoordinator.record_human_response(task, "reject")
    assert HumanPauseCoordinator.is_rejected(task) is True
    assert HumanPauseCoordinator.is_resumed(task) is False

    task2 = Task(tenant_id="t1", user_id="u1", message="x")
    HumanPauseCoordinator.record_human_response(task2, "escalate")
    assert HumanPauseCoordinator.is_escalated(task2) is True


def test_escalation_router_levels():
    router = EscalationRouter(max_levels=3)
    task = Task(tenant_id="t1", user_id="u1", message="x")

    first = router.route(task)
    assert first.target == EscalationTarget.HUMAN_OPERATOR
    router.apply_to_task(task, first)

    second = router.route(task)
    assert second.target == EscalationTarget.APPLICATION_ADMIN
    router.apply_to_task(task, second)

    third = router.route(task)
    assert third.fail_task is True
    assert third.target == EscalationTarget.FAIL_TASK


def test_human_decision_store_records_and_lists(tmp_path):
    store = SQLiteHumanDecisionStore(db_path=tmp_path / "human.db")
    record = SQLiteHumanDecisionStore.build_record(
        task_id="task-1",
        tenant_id="t1",
        user_id="u1",
        verdict=HumanResponseVerdict.ESCALATE,
        response_text="escalate",
        escalation_level=1,
        escalation_target=EscalationTarget.HUMAN_OPERATOR,
    )
    store.record(record)

    listed = store.list_for_task("task-1", "t1")
    assert len(listed) == 1
    assert listed[0].verdict == HumanResponseVerdict.ESCALATE

    escalations = store.list_escalations("t1")
    assert len(escalations) == 1
