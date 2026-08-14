# © Artur Czarnecki. All rights reserved.

"""Orchestration-owned declarative HITL grant lifecycle (ADR-PLATFORM-PLUGIN-001)."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from intergrax.contracts.declarative_hitl import (
    DeclarativeHitlApprovalGrant,
    DeclarativeHitlPendingApproval,
)
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import Task


class DeclarativeHitlGrantCoordinator:
    """Human/orchestration-side pending and grant state management."""

    @staticmethod
    def create_grant_from_pending(task: Task) -> DeclarativeHitlApprovalGrant | None:
        pending = task.runtime.governance.declarative_hitl_pending
        if pending is None:
            return None
        grant = DeclarativeHitlApprovalGrant(
            grant_id=f"grant_{uuid4().hex[:16]}",
            invocation_scope_id=pending.invocation_scope_id,
            task_id=pending.task_id,
            run_id=pending.run_id,
            step_id=pending.step_id,
            tool_id=pending.tool_id,
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
        request.declarative_hitl_grant = grant
        task.runtime.governance.declarative_hitl_grant = None
        task.sync_metadata()
        return request
