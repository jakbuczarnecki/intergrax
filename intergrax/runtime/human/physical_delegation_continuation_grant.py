# © Artur Czarnecki. All rights reserved.

"""Physical delegation governed continuation grant derivation and matching."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from intergrax.contracts.physical_delegation_governance import (
    PhysicalDelegationContinuationApprovalGrant,
    PhysicalDelegationGovernedContinuation,
    PhysicalDelegationGovernanceResult,
    grant_matches_physical_delegation_continuation,
    physical_delegation_governed_continuation_digest,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.task.task import Task

__all__ = [
    "PhysicalDelegationContinuationGrantCoordinator",
    "PhysicalDelegationContinuationGrantError",
    "matches_current_physical_delegation_requirement",
]


class PhysicalDelegationContinuationGrantError(ValueError):
    """Fail-closed physical delegation continuation grant handling."""


def matches_current_physical_delegation_requirement(
    grant: PhysicalDelegationContinuationApprovalGrant,
    *,
    continuation: PhysicalDelegationGovernedContinuation,
    current_result: PhysicalDelegationGovernanceResult,
) -> bool:
    """Pure fail-closed matcher — does not consume grant or execute delegation."""
    action = current_result.decision.action
    if action not in (PolicyAction.REQUIRE_HUMAN, PolicyAction.ALLOW):
        return False
    if not grant_matches_physical_delegation_continuation(grant, continuation):
        return False
    evidence = current_result.evidence
    grant_rule = grant.policy_rule_id.strip()
    current_rule = (
        evidence.policy_rule_id.strip()
        or current_result.decision.policy_rule_id.strip()
    )
    if not grant_rule or grant_rule != current_rule:
        return False
    if grant.governance_request_digest != evidence.request_digest:
        return False
    return True


class PhysicalDelegationContinuationGrantCoordinator:
    """Derive scoped physical delegation grant from canonical pause + approval."""

    @staticmethod
    def clear_grant(task: Task) -> None:
        task.runtime.governance.physical_delegation_continuation_grant = None

    @staticmethod
    def consume_matching_grant(
        task: Task,
        *,
        expected_grant_id: str,
    ) -> PhysicalDelegationContinuationApprovalGrant | None:
        gov = task.runtime.governance
        stored = gov.physical_delegation_continuation_grant
        if stored is None:
            return None
        if stored.grant_id != expected_grant_id:
            return None
        gov.physical_delegation_continuation_grant = None
        task.sync_metadata()
        return stored

    @staticmethod
    def _validate_resolution(task: Task) -> PhysicalDelegationGovernedContinuation:
        gov = task.runtime.governance
        resolution = gov.hitl_resolution
        if resolution is None:
            raise PhysicalDelegationContinuationGrantError(
                "canonical approval resolution required",
            )
        if resolution.verdict is not HumanResponseVerdict.APPROVE:
            raise PhysicalDelegationContinuationGrantError(
                "approval resolution verdict is not approve",
            )
        if resolution.task_id != task.task_id:
            raise PhysicalDelegationContinuationGrantError("resolution task_id mismatch")

        pause_record = gov.pause_record
        if pause_record is None:
            raise PhysicalDelegationContinuationGrantError("active pause record required")
        if resolution.pause_id != pause_record.pause_id:
            raise PhysicalDelegationContinuationGrantError("resolution pause_id mismatch")
        if resolution.human_request_id != pause_record.human_request_id:
            raise PhysicalDelegationContinuationGrantError(
                "resolution human_request_id mismatch",
            )

        human_request = gov.human_request
        if human_request is None:
            raise PhysicalDelegationContinuationGrantError("active human request required")
        if human_request.request_id != pause_record.human_request_id:
            raise PhysicalDelegationContinuationGrantError("human_request identity mismatch")
        if human_request.governed_continuation is None:
            raise PhysicalDelegationContinuationGrantError(
                "governed continuation correlation required",
            )

        continuation = gov.physical_delegation_governed_continuation
        if continuation is None:
            raise PhysicalDelegationContinuationGrantError(
                "physical delegation governed continuation required",
            )
        if continuation.task_scope_id != task.task_id:
            raise PhysicalDelegationContinuationGrantError(
                "continuation task_scope_id mismatch",
            )

        correlation = human_request.governed_continuation
        if correlation.task_id != continuation.task_scope_id:
            raise PhysicalDelegationContinuationGrantError(
                "continuation task_id mismatch",
            )
        if resolution.run_id is None:
            raise PhysicalDelegationContinuationGrantError("resolution run_id required")
        if resolution.run_id != correlation.run_id:
            raise PhysicalDelegationContinuationGrantError("continuation run_id mismatch")
        return continuation

    @staticmethod
    def create_grant_from_approval(
        task: Task,
    ) -> PhysicalDelegationContinuationApprovalGrant | None:
        gov = task.runtime.governance
        human_request = gov.human_request
        if human_request is None or human_request.governed_continuation is None:
            return None
        if gov.physical_delegation_governed_continuation is None:
            return None
        resolution = gov.hitl_resolution
        if resolution is None or resolution.verdict is not HumanResponseVerdict.APPROVE:
            return None

        continuation = PhysicalDelegationContinuationGrantCoordinator._validate_resolution(
            task,
        )
        pause_record = gov.pause_record
        correlation = human_request.governed_continuation
        assert pause_record is not None

        evidence = continuation.governance_result.evidence
        grant = PhysicalDelegationContinuationApprovalGrant(
            grant_id=f"pdcg_{uuid4().hex[:16]}",
            continuation_digest=physical_delegation_governed_continuation_digest(
                continuation,
            ),
            continuation_request_id=correlation.continuation_request_id,
            delegation_id=continuation.delegation_id,
            task_scope_id=continuation.task_scope_id,
            run_id=correlation.run_id,
            selected_identity=continuation.selected_identity,
            capability_requirement=continuation.capability_requirement,
            governance_request_digest=evidence.request_digest,
            policy_rule_id=evidence.policy_rule_id or continuation.governance_result.decision.policy_rule_id,
            policy_decision_id=evidence.policy_decision_id,
            pause_id=resolution.pause_id,
            human_request_id=resolution.human_request_id,
            approved_at=datetime.now(timezone.utc).isoformat(),
        )
        gov.physical_delegation_continuation_grant = grant
        task.sync_metadata()
        return grant
