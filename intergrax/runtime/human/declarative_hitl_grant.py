# © Artur Czarnecki. All rights reserved.

"""Orchestration-owned declarative HITL grant lifecycle (ADR-PLATFORM-PLUGIN-001)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4

from intergrax.contracts.declarative_hitl import (
    DeclarativeHitlApprovalGrant,
    DeclarativeHitlPendingApproval,
)
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import Task


class DeclarativeHitlGrantError(ValueError):
    """Fail-closed declarative HITL grant creation without canonical approval."""


class DeclarativeHitlGrantCoordinator:
    """Human/orchestration-side pending and grant state management."""

    @staticmethod
    def _validate_resolution_for_pending(task: Task, pending: DeclarativeHitlPendingApproval) -> None:
        resolution = task.runtime.governance.hitl_resolution
        if resolution is None:
            raise DeclarativeHitlGrantError("canonical approval resolution required")
        if resolution.verdict is not HumanResponseVerdict.APPROVE:
            raise DeclarativeHitlGrantError("approval resolution verdict is not approve")
        if resolution.task_id != pending.task_id:
            raise DeclarativeHitlGrantError("resolution task_id mismatch")
        if resolution.pause_id != pending.pause_id:
            raise DeclarativeHitlGrantError("resolution pause_id mismatch")
        if resolution.human_request_id != pending.human_request_id:
            raise DeclarativeHitlGrantError("resolution human_request_id mismatch")

    @staticmethod
    def create_grant_from_pending(task: Task) -> DeclarativeHitlApprovalGrant | None:
        pending = task.runtime.governance.declarative_hitl_pending
        if pending is None:
            return None
        DeclarativeHitlGrantCoordinator._validate_resolution_for_pending(task, pending)
        grant = DeclarativeHitlApprovalGrant(
            grant_id=f"grant_{uuid4().hex[:16]}",
            invocation_scope_id=pending.invocation_scope_id,
            task_id=pending.task_id,
            run_id=pending.run_id,
            step_id=pending.step_id,
            tool_id=pending.tool_id,
            agent_id=pending.agent_id,
            idempotency_key=pending.idempotency_key,
            matched_rule_ids=pending.matched_rule_ids,
            human_request_id=pending.human_request_id,
            policy_provenance_digest=pending.policy_provenance_digest,
            pause_id=pending.pause_id,
            approved_at=datetime.now(timezone.utc).isoformat(),
        )
        task.runtime.governance.declarative_hitl_grant = grant
        task.runtime.governance.declarative_hitl_pending = None
        return grant

    @staticmethod
    def clear_pending_and_grant(task: Task) -> None:
        task.runtime.governance.declarative_hitl_pending = None
        task.runtime.governance.declarative_hitl_grant = None

    @staticmethod
    def transfer_persisted_grant_for_resume(task: Task, request: RuntimeRequest) -> RuntimeRequest:
        """Consume persisted grant at orchestration resume boundary."""
        grant = task.runtime.governance.declarative_hitl_grant
        if grant is None:
            return request
        updated = replace(request, declarative_hitl_grant=grant)
        if updated.task_id is None:
            updated = replace(updated, task_id=task.task_id)
        task.runtime.governance.declarative_hitl_grant = None
        task.sync_metadata()
        return updated
