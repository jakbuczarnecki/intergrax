# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.contracts.agent_decision import HumanRequest
from intergrax.contracts.declarative_hitl import DeclarativeHitlPendingApproval
from intergrax.runtime.human.declarative_hitl_grant import DeclarativeHitlGrantCoordinator
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_contract import TaskPauseRecord

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _approve_resolution(task: Task, *, pause_id: str = "pause-1", human_request_id: str = "hr-1") -> None:
    task.runtime.governance.paused = True
    task.runtime.governance.pause_record = TaskPauseRecord(
        pause_id=pause_id,
        task_id=task.task_id,
        human_request_id=human_request_id,
    )
    task.runtime.governance.human_request = HumanRequest(
        request_id=human_request_id,
        prompt="approve?",
    )
    HumanPauseCoordinator.resolve_human_response(
        task,
        HumanResponseVerdict.APPROVE,
        approver=local_development_approver_evidence(tenant_id=task.tenant_id),
        pause_id=pause_id,
        human_request_id=human_request_id,
    )


def _pending(task_id: str = "task_test01") -> DeclarativeHitlPendingApproval:
    return DeclarativeHitlPendingApproval(
        invocation_scope_id="dhr_scope",
        task_id=task_id,
        run_id="run-1",
        step_id="step-1",
        tool_id="tool.a",
        idempotency_key=None,
        matched_rule_ids=("rule-1",),
        human_request_id="hr-1",
        policy_provenance_digest="digest-1",
        agent_id="agent-1",
        pause_id="pause-1",
        created_at="2026-08-14T00:00:00+00:00",
    )


def test_create_grant_from_pending_copies_scope() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x")
    task.runtime.governance.declarative_hitl_pending = _pending(task.task_id)
    _approve_resolution(task)
    grant = DeclarativeHitlGrantCoordinator.create_grant_from_pending(task)
    assert grant is not None
    assert grant.invocation_scope_id == "dhr_scope"
    assert grant.agent_id == "agent-1"
    assert task.runtime.governance.declarative_hitl_pending is None
    assert task.runtime.governance.declarative_hitl_grant is grant


def test_no_grant_without_pending() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x")
    assert DeclarativeHitlGrantCoordinator.create_grant_from_pending(task) is None


def test_clear_pending_and_grant() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x")
    task.runtime.governance.declarative_hitl_pending = _pending(task.task_id)
    _approve_resolution(task)
    DeclarativeHitlGrantCoordinator.create_grant_from_pending(task)
    DeclarativeHitlGrantCoordinator.clear_pending_and_grant(task)
    assert task.runtime.governance.declarative_hitl_pending is None
    assert task.runtime.governance.declarative_hitl_grant is None


def test_transfer_persisted_grant_for_resume() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x")
    task.runtime.governance.declarative_hitl_pending = _pending(task.task_id)
    _approve_resolution(task)
    grant = DeclarativeHitlGrantCoordinator.create_grant_from_pending(task)
    request = RuntimeRequest(
        agent_id="agent-1",
        user_id="u1",
        session_id="s1",
        message="x",
        task_id=task.task_id,
        run_id=mint_run_id(),
    )
    updated = DeclarativeHitlGrantCoordinator.transfer_persisted_grant_for_resume(task, request)
    assert updated.declarative_hitl_grant == grant
    assert updated.task_id == task.task_id
    assert task.runtime.governance.declarative_hitl_grant is None
