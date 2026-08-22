# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.execution_identity import mint_task_id
from intergrax.runtime.human.escalation import EscalationRouter, parse_human_response
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.runtime.human.models import (
    EscalationTarget,
    HumanDecisionRecord,
    HumanResponseVerdict,
    build_human_decision_record,
)
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.human.persistence_contract import (
    HumanDecisionPersistence,
    InMemoryHumanDecisionPersistence,
)
from intergrax.runtime.human.store import SQLiteHumanDecisionStore
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.orchestration.human_response import persist_human_decision
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_contract import TaskPauseRecord

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


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
    assert isinstance(store, HumanDecisionPersistence)
    record = build_human_decision_record(
        task_id="task-1",
        tenant_id="t1",
        approver=local_development_approver_evidence(tenant_id="t1", actor_id="u1"),
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


def test_persist_human_decision_uses_generic_persistence() -> None:
    store = InMemoryHumanDecisionPersistence()
    task_id = mint_task_id()
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=task_id)
    task.runtime.governance.paused = True
    task.runtime.governance.pause_record = TaskPauseRecord(
        pause_id="pause-1",
        task_id=task_id,
        human_request_id="hr-1",
    )
    approver = local_development_approver_evidence(tenant_id="t1")
    HumanPauseCoordinator.resolve_human_response(
        task,
        HumanResponseVerdict.APPROVE,
        approver=approver,
        pause_id="pause-1",
        human_request_id="hr-1",
        response_text="approve",
    )

    persist_human_decision(
        task,
        HumanResponseVerdict.APPROVE,
        human_store=store,
        response_text="approve",
    )

    listed = store.list_for_task(task_id, "t1")
    assert len(listed) == 1
    assert isinstance(listed[0], HumanDecisionRecord)
    assert listed[0].verdict is HumanResponseVerdict.APPROVE
    assert listed[0].response_text == "approve"
    assert listed[0].approver.user_id == approver.user_id


def test_nexus_loop_accepts_human_decision_persistence() -> None:
    store = InMemoryHumanDecisionPersistence()
    loop = NexusLoop(AgentRegistry(), human_decision_store=store)
    assert loop._human_store is store
