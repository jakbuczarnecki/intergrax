# © Artur Czarnecki. All rights reserved.

"""Orchestration-owned governed continuation grant derivation (G5C-2B-1)."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from intergrax.contracts.governed_continuation_grant import GovernedContinuationApprovalGrant
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.task.task import Task


class GovernedContinuationGrantError(ValueError):
    """Fail-closed governed continuation grant creation without canonical approval."""


class GovernedContinuationGrantCoordinator:
    """Derive scoped continuation grant from canonical pause + approval — no global state."""

    @staticmethod
    def clear_grant(task: Task) -> None:
        task.runtime.governance.governed_continuation_grant = None

    @staticmethod
    def _validate_resolution(task: Task) -> None:
        gov = task.runtime.governance
        resolution = gov.hitl_resolution
        if resolution is None:
            raise GovernedContinuationGrantError("canonical approval resolution required")
        if resolution.verdict is not HumanResponseVerdict.APPROVE:
            raise GovernedContinuationGrantError("approval resolution verdict is not approve")
        if resolution.task_id != task.task_id:
            raise GovernedContinuationGrantError("resolution task_id mismatch")

        pause_record = gov.pause_record
        if pause_record is None:
            raise GovernedContinuationGrantError("active pause record required")
        if resolution.pause_id != pause_record.pause_id:
            raise GovernedContinuationGrantError("resolution pause_id mismatch")
        if resolution.human_request_id != pause_record.human_request_id:
            raise GovernedContinuationGrantError("resolution human_request_id mismatch")

        human_request = gov.human_request
        if human_request is None:
            raise GovernedContinuationGrantError("active human request required")
        if human_request.request_id != pause_record.human_request_id:
            raise GovernedContinuationGrantError("human_request identity mismatch")
        if human_request.governed_continuation is None:
            raise GovernedContinuationGrantError("governed continuation correlation required")

        continuation = human_request.governed_continuation
        if not continuation.task_id:
            raise GovernedContinuationGrantError("continuation task_id required")
        if not continuation.run_id:
            raise GovernedContinuationGrantError("continuation run_id required")
        if continuation.task_id != task.task_id:
            raise GovernedContinuationGrantError("continuation task_id mismatch")
        if resolution.task_id != continuation.task_id:
            raise GovernedContinuationGrantError("resolution continuation task_id mismatch")

        if resolution.run_id is None:
            raise GovernedContinuationGrantError("resolution run_id required")
        if resolution.run_id != continuation.run_id:
            raise GovernedContinuationGrantError("continuation run_id mismatch")

    @staticmethod
    def create_grant_from_approval(task: Task) -> GovernedContinuationApprovalGrant | None:
        gov = task.runtime.governance
        continuation = (
            gov.human_request.governed_continuation
            if gov.human_request is not None
            else None
        )
        if continuation is None:
            return None
        if continuation.side_effect_scope_id is None:
            return None

        GovernedContinuationGrantCoordinator._validate_resolution(task)
        resolution = gov.hitl_resolution
        pause_record = gov.pause_record
        assert resolution is not None
        assert pause_record is not None

        grant = GovernedContinuationApprovalGrant(
            grant_id=f"gcg_{uuid4().hex[:16]}",
            continuation_request_id=continuation.continuation_request_id,
            side_effect_scope_id=continuation.side_effect_scope_id,
            side_effect_scope_digest=continuation.side_effect_scope_digest,
            task_id=continuation.task_id,
            run_id=continuation.run_id,
            operation_id=continuation.operation_id,
            resource_scope=continuation.resource_scope,
            policy_rule_id=continuation.policy_rule_id,
            pause_id=resolution.pause_id,
            human_request_id=resolution.human_request_id,
            approved_at=datetime.now(timezone.utc).isoformat(),
        )
        gov.governed_continuation_grant = grant
        return grant
