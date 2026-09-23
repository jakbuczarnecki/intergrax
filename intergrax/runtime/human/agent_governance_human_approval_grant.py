# © Artur Czarnecki. All rights reserved.

"""Human APPROVE → durable Agent Governance grant (UCA-6C-R6-R5.6)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from intergrax.contracts.agent_governance_grant_lifecycle_port import (
    AgentGovernanceGrantLifecycleMutationResult,
    AgentGovernanceGrantLifecycleOutcome,
)
from intergrax.contracts.agent_governance_hitl import (
    AgentGovernanceGrantLifecycleState,
    AgentGovernanceHumanApprovalGrant,
    AgentGovernanceHumanApprovalPending,
)
from intergrax.contracts.human_approver import HumanApproverEvidence
from intergrax.runtime.human.agent_governance_grant_lifecycle import (
    TaskAgentGovernanceGrantLifecycleAdapter,
)
from intergrax.runtime.human.agent_governance_pause_projection import (
    AgentGovernancePauseProjectionOutcome,
    TaskAgentGovernancePauseProjectionAdapter,
)
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.long_running.persistence_contract import (
    TaskCheckpointPersistence,
)
from intergrax.runtime.task.task import Task

_AGENT_GRANT_TTL = timedelta(hours=24)


class AgentGovernanceHumanApprovalGrantError(ValueError):
    """Fail-closed Agent Governance grant materialization."""


class AgentGovernanceHumanApprovalGrantCoordinator:
    """Orchestration-side grant minting from canonical pending + Task checkpoint CAS."""

    @staticmethod
    def _validate_resolution_for_pending(
        task: Task,
        pending: AgentGovernanceHumanApprovalPending,
    ) -> None:
        resolution = task.runtime.governance.hitl_resolution
        if resolution is None:
            raise AgentGovernanceHumanApprovalGrantError(
                "canonical approval resolution required",
            )
        if resolution.verdict is not HumanResponseVerdict.APPROVE:
            raise AgentGovernanceHumanApprovalGrantError(
                "approval resolution verdict is not approve",
            )
        if str(resolution.task_id) != str(pending.task_id):
            raise AgentGovernanceHumanApprovalGrantError("resolution task_id mismatch")
        if resolution.pause_id != pending.pause_id:
            raise AgentGovernanceHumanApprovalGrantError("resolution pause_id mismatch")
        if resolution.human_request_id != pending.human_request_id:
            raise AgentGovernanceHumanApprovalGrantError(
                "resolution human_request_id mismatch",
            )

    @staticmethod
    def _grant_matches_pending(
        grant: AgentGovernanceHumanApprovalGrant,
        pending: AgentGovernanceHumanApprovalPending,
    ) -> bool:
        requirement = pending.requirement
        return (
            grant.agent_governance_invocation_scope_id
            == pending.agent_governance_invocation_scope_id
            and grant.pending_generation == pending.generation
            and grant.logical_invocation_fingerprint
            == requirement.logical_invocation_fingerprint
            and grant.task_id == pending.task_id
            and grant.run_id == pending.run_id
            and grant.attempt_id == pending.attempt_id
            and grant.execution_id == pending.execution_id
            and grant.human_request_id == pending.human_request_id
            and grant.pause_id == pending.pause_id
        )

    @staticmethod
    def _mint_grant(
        pending: AgentGovernanceHumanApprovalPending,
        *,
        approver: HumanApproverEvidence,
    ) -> AgentGovernanceHumanApprovalGrant:
        requirement = pending.requirement
        now = datetime.now(timezone.utc)
        expires = now + _AGENT_GRANT_TTL
        return AgentGovernanceHumanApprovalGrant(
            grant_id=f"agrg_{uuid4().hex[:16]}",
            agent_governance_invocation_scope_id=pending.agent_governance_invocation_scope_id,
            pending_generation=pending.generation,
            logical_invocation_fingerprint=requirement.logical_invocation_fingerprint,
            task_id=pending.task_id,
            run_id=pending.run_id,
            attempt_id=pending.attempt_id,
            execution_id=pending.execution_id,
            tenant_id=pending.tenant_id,
            agent_id=pending.agent_id,
            tool_id=pending.tool_id,
            step_id=pending.step_id,
            idempotency_key=pending.idempotency_key,
            policy_provenance_digest=pending.policy_provenance_digest,
            human_request_id=pending.human_request_id,
            pause_id=pending.pause_id,
            approved_at=now.isoformat(),
            expires_at=expires.isoformat(),
            decided_by=approver.user_id,
        )

    @staticmethod
    def persist_available_grant_from_human_approve(
        task: Task,
        *,
        checkpoint_store: TaskCheckpointPersistence,
        approver: HumanApproverEvidence,
    ) -> AgentGovernanceGrantLifecycleMutationResult:
        pending = task.runtime.governance.agent_governance_hitl_pending
        if pending is None:
            return AgentGovernanceGrantLifecycleMutationResult(
                outcome=AgentGovernanceGrantLifecycleOutcome.NOT_FOUND,
            )
        AgentGovernanceHumanApprovalGrantCoordinator._validate_resolution_for_pending(
            task,
            pending,
        )
        adapter = TaskAgentGovernanceGrantLifecycleAdapter(
            task=task,
            checkpoint_store=checkpoint_store,
        )
        current = adapter.load(task_id=task.task_id, tenant_id=task.tenant_id)
        if current is not None:
            if AgentGovernanceHumanApprovalGrantCoordinator._grant_matches_pending(
                current.grant,
                pending,
            ):
                if current.lifecycle_state in {
                    AgentGovernanceGrantLifecycleState.AVAILABLE,
                    AgentGovernanceGrantLifecycleState.RESERVED,
                    AgentGovernanceGrantLifecycleState.APPLIED,
                }:
                    return AgentGovernanceGrantLifecycleMutationResult(
                        outcome=AgentGovernanceGrantLifecycleOutcome.APPLIED,
                        record=current,
                    )
            raise AgentGovernanceHumanApprovalGrantError(
                "incompatible agent governance grant already present",
            )
        grant = AgentGovernanceHumanApprovalGrantCoordinator._mint_grant(
            pending,
            approver=approver,
        )
        expected_revision = current.lifecycle_revision if current is not None else 0
        result = adapter.persist_available_grant(
            task_id=task.task_id,
            tenant_id=task.tenant_id,
            expected_lifecycle_revision=expected_revision,
            grant=grant,
        )
        if result.outcome is not AgentGovernanceGrantLifecycleOutcome.APPLIED:
            return result
        task.sync_metadata()
        return result

    @staticmethod
    def clear_pending_on_reject_or_escalate(
        task: Task,
        *,
        checkpoint_store: TaskCheckpointPersistence,
    ) -> None:
        if task.runtime.governance.agent_governance_hitl_pending is None:
            return
        adapter = TaskAgentGovernancePauseProjectionAdapter(
            task=task,
            checkpoint_store=checkpoint_store,
        )
        result = adapter.clear_pending_durably()
        if result.outcome is AgentGovernancePauseProjectionOutcome.STALE_REVISION:
            raise AgentGovernanceHumanApprovalGrantError(
                "stale checkpoint while clearing agent governance pending",
            )
        if result.outcome is not AgentGovernancePauseProjectionOutcome.APPLIED:
            raise AgentGovernanceHumanApprovalGrantError(
                "failed to clear agent governance pending durably",
            )

    @staticmethod
    def terminalize_after_successful_consumption(
        task: Task,
        *,
        checkpoint_store: TaskCheckpointPersistence,
        reason: str = "suspended_work_consumed",
    ) -> AgentGovernanceGrantLifecycleMutationResult:
        record = task.runtime.governance.agent_governance_human_approval_grant
        if record is None:
            return AgentGovernanceGrantLifecycleMutationResult(
                outcome=AgentGovernanceGrantLifecycleOutcome.NOT_FOUND,
            )
        if record.lifecycle_state is not AgentGovernanceGrantLifecycleState.APPLIED:
            return AgentGovernanceGrantLifecycleMutationResult(
                outcome=AgentGovernanceGrantLifecycleOutcome.INVALID_STATE,
                record=record,
            )
        adapter = TaskAgentGovernanceGrantLifecycleAdapter(
            task=task,
            checkpoint_store=checkpoint_store,
        )
        result = adapter.terminalize(
            task_id=task.task_id,
            tenant_id=task.tenant_id,
            expected_lifecycle_revision=record.lifecycle_revision,
            reason=reason,
        )
        if result.outcome is AgentGovernanceGrantLifecycleOutcome.APPLIED:
            task.sync_metadata()
        return result


__all__ = [
    "AgentGovernanceHumanApprovalGrantCoordinator",
    "AgentGovernanceHumanApprovalGrantError",
]
