# © Artur Czarnecki. All rights reserved.

"""Agent Governance grant reserve/verify for suspended work re-entry (UCA-6C-R6-R5.6)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from intergrax.contracts.agent_governance_grant_lifecycle_port import (
    AgentGovernanceGrantLifecycleOutcome,
)
from intergrax.contracts.agent_governance_hitl import (
    AgentGovernanceGrantLifecycleRecord,
    AgentGovernanceGrantLifecycleState,
)
from intergrax.contracts.agent_governance_verified_approval import (
    VerifiedAgentGovernanceHumanApproval,
)
from intergrax.contracts.agent_runtime_governance import ToolAuthorizationRequest
from intergrax.contracts.execution.suspended_operation.payload_catalog import (
    ExecutionBoundCatalogToolOperationPayload,
)
from intergrax.contracts.lease_claim import LeaseOwnership
from intergrax.runtime.agent_governance.grant_verifier import (
    AgentGovernanceGrantVerificationError,
    AgentGovernanceGrantVerifier,
)
from intergrax.runtime.agent_governance.request_builder import (
    build_tool_authorization_request,
    governance_capability_for_contract,
)
from intergrax.runtime.human.agent_governance_grant_lifecycle import (
    TaskAgentGovernanceGrantLifecycleAdapter,
)
from intergrax.runtime.long_running.persistence_contract import (
    TaskCheckpointPersistence,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.task.task import Task
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest


@dataclass(frozen=True, slots=True)
class AgentGovernanceReentryGrantPrepareResult:
    verified: VerifiedAgentGovernanceHumanApproval
    lifecycle_record: AgentGovernanceGrantLifecycleRecord
    claim_ownership: LeaseOwnership


class AgentGovernanceReentryGrantError(RuntimeError):
    """Fail-closed grant preparation for re-entry."""


def is_agent_governance_invocation_scope(invocation_scope_id: str) -> bool:
    return invocation_scope_id.startswith("agr_")


def build_authorization_request_for_catalog_payload(
    *,
    payload: ExecutionBoundCatalogToolOperationPayload,
    contract: ToolContract,
    state: RuntimeState,
    agent_id: str,
) -> ToolAuthorizationRequest:
    invocation_context = None
    tool_request = ToolExecutionRequest(
        run_id=payload.run_id,
        step_id=payload.step_id,
        tool_id=payload.tool_id,
        input=payload.tool_input,
        invocation_context=invocation_context,
        idempotency_key=payload.idempotency_key,
        declarative_hitl_invocation_scope_id=None,
    )
    return build_tool_authorization_request(
        state=state,
        agent_id=agent_id,
        contract=contract,
        request=tool_request,
    )


def prepare_agent_governance_grant_for_reentry(
    *,
    task: Task,
    checkpoint_store: TaskCheckpointPersistence,
    payload: ExecutionBoundCatalogToolOperationPayload,
    contract: ToolContract,
    state: RuntimeState,
    claim_ownership: LeaseOwnership,
    pause_generation: int,
    lease_seconds: int,
    verifier: AgentGovernanceGrantVerifier | None = None,
) -> AgentGovernanceReentryGrantPrepareResult:
    pending = task.runtime.governance.agent_governance_hitl_pending
    adapter = TaskAgentGovernanceGrantLifecycleAdapter(
        task=task,
        checkpoint_store=checkpoint_store,
    )
    record = adapter.load(task_id=task.task_id, tenant_id=task.tenant_id)
    if record is None:
        raise AgentGovernanceReentryGrantError("agent governance grant missing")

    auth_request = build_authorization_request_for_catalog_payload(
        payload=payload,
        contract=contract,
        state=state,
        agent_id=payload.agent_id,
    )
    _ = governance_capability_for_contract(contract)

    if pending is not None:
        requirement = pending.requirement
    elif record.lifecycle_state is AgentGovernanceGrantLifecycleState.APPLIED:
        from intergrax.contracts.agent_governance_hitl import (
            AgentGovernanceHumanApprovalRequirement,
        )

        requirement = AgentGovernanceHumanApprovalRequirement(
            agent_governance_invocation_scope_id=record.grant.agent_governance_invocation_scope_id,
            task_id=record.grant.task_id,
            run_id=record.grant.run_id,
            attempt_id=record.grant.attempt_id,
            execution_id=record.grant.execution_id,
            tenant_id=record.grant.tenant_id,
            agent_id=record.grant.agent_id,
            tool_id=record.grant.tool_id,
            step_id=record.grant.step_id,
            idempotency_key=record.grant.idempotency_key,
            approval_id=record.grant.human_request_id,
            authorization_request=auth_request,
            logical_invocation_fingerprint=record.grant.logical_invocation_fingerprint,
            pause_generation=pause_generation,
            policy_provenance_digest=record.grant.policy_provenance_digest,
        )
    else:
        raise AgentGovernanceReentryGrantError(
            "agent governance pending required before grant consumption",
        )

    if record.lifecycle_state is AgentGovernanceGrantLifecycleState.AVAILABLE:
        lease_expires = datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
        reserved = adapter.reserve(
            task_id=task.task_id,
            tenant_id=task.tenant_id,
            expected_lifecycle_revision=record.lifecycle_revision,
            grant_id=record.grant.grant_id,
            logical_invocation_fingerprint=record.grant.logical_invocation_fingerprint,
            pause_generation=pause_generation,
            agent_governance_invocation_scope_id=record.grant.agent_governance_invocation_scope_id,
            task_id_link=record.grant.task_id,
            run_id=record.grant.run_id,
            attempt_id=record.grant.attempt_id,
            execution_id=record.grant.execution_id,
            policy_provenance_digest=record.grant.policy_provenance_digest,
            owner_id=claim_ownership.owner_id,
            lease_expires_at=lease_expires,
        )
        if reserved.outcome is not AgentGovernanceGrantLifecycleOutcome.APPLIED:
            raise AgentGovernanceReentryGrantError(
                f"grant reserve failed: {reserved.outcome.value}",
            )
        record = reserved.record
        if record is None:
            raise AgentGovernanceReentryGrantError("grant reserve missing record")
    elif record.lifecycle_state is AgentGovernanceGrantLifecycleState.RESERVED:
        reservation = record.reservation
        if reservation is None:
            raise AgentGovernanceReentryGrantError("reserved grant missing reservation")
        ownership = reservation.ownership
        if (
            ownership.owner_id != claim_ownership.owner_id
            or ownership.fence != claim_ownership.fence
        ):
            raise AgentGovernanceReentryGrantError("grant reservation owner mismatch")
        if ownership.lease_expires_at <= datetime.now(timezone.utc):
            raise AgentGovernanceReentryGrantError("grant reservation lease expired")
    elif record.lifecycle_state is not AgentGovernanceGrantLifecycleState.APPLIED:
        raise AgentGovernanceReentryGrantError(
            f"grant lifecycle invalid for reentry: {record.lifecycle_state.value}",
        )

    active_verifier = verifier or AgentGovernanceGrantVerifier()
    try:
        verified = active_verifier.verify_for_resume(
            lifecycle_record=record,
            pending=pending,
            requirement=requirement,
            request=auth_request,
            logical_invocation_fingerprint=record.grant.logical_invocation_fingerprint,
            pause_generation=pause_generation,
        )
    except AgentGovernanceGrantVerificationError as exc:
        raise AgentGovernanceReentryGrantError(str(exc)) from exc

    reservation_ownership = (
        record.reservation.ownership
        if record.reservation is not None
        else claim_ownership
    )
    return AgentGovernanceReentryGrantPrepareResult(
        verified=verified,
        lifecycle_record=record,
        claim_ownership=reservation_ownership,
    )


def mark_agent_governance_grant_applied_after_governance(
    *,
    task: Task,
    checkpoint_store: TaskCheckpointPersistence,
    lifecycle_record: AgentGovernanceGrantLifecycleRecord,
    claim_ownership: LeaseOwnership,
) -> None:
    if lifecycle_record.lifecycle_state is AgentGovernanceGrantLifecycleState.APPLIED:
        return
    adapter = TaskAgentGovernanceGrantLifecycleAdapter(
        task=task,
        checkpoint_store=checkpoint_store,
    )
    applied = adapter.mark_applied(
        task_id=task.task_id,
        tenant_id=task.tenant_id,
        expected_lifecycle_revision=lifecycle_record.lifecycle_revision,
        owner_id=claim_ownership.owner_id,
        fence=claim_ownership.fence,
    )
    if applied.outcome is not AgentGovernanceGrantLifecycleOutcome.APPLIED:
        raise AgentGovernanceReentryGrantError(
            f"grant mark_applied failed: {applied.outcome.value}",
        )
    task.sync_metadata()


__all__ = [
    "AgentGovernanceReentryGrantError",
    "AgentGovernanceReentryGrantPrepareResult",
    "build_authorization_request_for_catalog_payload",
    "is_agent_governance_invocation_scope",
    "mark_agent_governance_grant_applied_after_governance",
    "prepare_agent_governance_grant_for_reentry",
]
