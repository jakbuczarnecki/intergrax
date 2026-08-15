# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.declarative_hitl import DeclarativeHitlPendingApproval
from intergrax.runtime.human.declarative_hitl_grant import DeclarativeHitlGrantCoordinator
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import Task

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _pending() -> DeclarativeHitlPendingApproval:
    return DeclarativeHitlPendingApproval(
        invocation_scope_id="dhr_scope",
        task_id="task-1",
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
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id="task-1")
    task.runtime.governance.declarative_hitl_pending = _pending()
    grant = DeclarativeHitlGrantCoordinator.create_grant_from_pending(task)
    assert grant is not None
    assert grant.invocation_scope_id == "dhr_scope"
    assert task.runtime.governance.declarative_hitl_pending is None
    assert task.runtime.governance.declarative_hitl_grant is grant


def test_no_grant_without_pending() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x")
    assert DeclarativeHitlGrantCoordinator.create_grant_from_pending(task) is None


def test_clear_pending_and_grant() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x")
    task.runtime.governance.declarative_hitl_pending = _pending()
    DeclarativeHitlGrantCoordinator.create_grant_from_pending(task)
    DeclarativeHitlGrantCoordinator.clear_pending_and_grant(task)
    assert task.runtime.governance.declarative_hitl_pending is None
    assert task.runtime.governance.declarative_hitl_grant is None


def test_transfer_persisted_grant_for_resume() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id="task-1")
    task.runtime.governance.declarative_hitl_pending = _pending()
    grant = DeclarativeHitlGrantCoordinator.create_grant_from_pending(task)
    request = RuntimeRequest(agent_id="agent-1", user_id="u1", session_id="s1", message="x")
    updated = DeclarativeHitlGrantCoordinator.transfer_persisted_grant_for_resume(task, request)
    assert updated.declarative_hitl_grant == grant
    assert updated.task_id == "task-1"
    assert task.runtime.governance.declarative_hitl_grant is None
