# © Artur Czarnecki. All rights reserved.

"""Agent Governance grant verifier — sole mint authority for verified approval tokens."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from intergrax.contracts.agent_governance_hitl import (
    AgentGovernanceGrantLifecycleRecord,
    AgentGovernanceGrantLifecycleState,
    AgentGovernanceHumanApprovalGrant,
    AgentGovernanceHumanApprovalPending,
    AgentGovernanceHumanApprovalRequirement,
    LogicalInvocationFingerprint,
)
from intergrax.contracts.agent_governance_verified_approval import (
    VerifiedAgentGovernanceHumanApproval,
)
from intergrax.contracts.agent_runtime_governance import ToolAuthorizationRequest
from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant
from intergrax.contracts.governed_continuation_grant import (
    GovernedContinuationApprovalGrant,
)


class AgentGovernanceGrantVerificationError(ValueError):
    """Fail-closed grant verification."""


@dataclass(frozen=True, slots=True)
class AgentGovernanceGrantVerifier:
    """Agent Runtime Governance-owned verifier (ADR-UCA-6C-ADR3-R2)."""

    def verify_for_resume(
        self,
        *,
        lifecycle_record: AgentGovernanceGrantLifecycleRecord,
        pending: AgentGovernanceHumanApprovalPending | None,
        requirement: AgentGovernanceHumanApprovalRequirement,
        request: ToolAuthorizationRequest,
        logical_invocation_fingerprint: LogicalInvocationFingerprint,
        pause_generation: int,
    ) -> VerifiedAgentGovernanceHumanApproval:
        grant = lifecycle_record.grant
        if lifecycle_record.lifecycle_state not in {
            AgentGovernanceGrantLifecycleState.AVAILABLE,
            AgentGovernanceGrantLifecycleState.RESERVED,
            AgentGovernanceGrantLifecycleState.APPLIED,
        }:
            raise AgentGovernanceGrantVerificationError("grant lifecycle terminal")
        if requirement.pause_generation != pause_generation:
            raise AgentGovernanceGrantVerificationError(
                "requirement generation mismatch"
            )
        self._verify_requirement_linkage(
            requirement=requirement,
            request=request,
            logical_invocation_fingerprint=logical_invocation_fingerprint,
            pause_generation=pause_generation,
        )
        if lifecycle_record.lifecycle_state in {
            AgentGovernanceGrantLifecycleState.AVAILABLE,
            AgentGovernanceGrantLifecycleState.RESERVED,
        }:
            if pending is None:
                raise AgentGovernanceGrantVerificationError(
                    "pending required before grant application"
                )
            self._verify_pending_grant_linkage(
                grant=grant,
                pending=pending,
                requirement=requirement,
                logical_invocation_fingerprint=logical_invocation_fingerprint,
                pause_generation=pause_generation,
            )
        elif pending is not None:
            self._verify_pending_grant_linkage(
                grant=grant,
                pending=pending,
                requirement=requirement,
                logical_invocation_fingerprint=logical_invocation_fingerprint,
                pause_generation=pause_generation,
            )
        else:
            self._verify_applied_recovery_linkage(
                grant=grant,
                requirement=requirement,
                logical_invocation_fingerprint=logical_invocation_fingerprint,
                pause_generation=pause_generation,
            )
        expires = datetime.fromisoformat(grant.expires_at.replace("Z", "+00:00"))
        if expires <= datetime.now(timezone.utc):
            raise AgentGovernanceGrantVerificationError("grant expired")
        return VerifiedAgentGovernanceHumanApproval(
            verification_id=f"vagr_{uuid4().hex[:16]}",
            grant_id=grant.grant_id,
            agent_governance_invocation_scope_id=grant.agent_governance_invocation_scope_id,
            requirement=requirement,
            grant=grant,
            logical_invocation_fingerprint=logical_invocation_fingerprint,
            task_id=grant.task_id,
            run_id=grant.run_id,
            attempt_id=grant.attempt_id,
            execution_id=grant.execution_id,
            pause_generation=pause_generation,
        )

    @staticmethod
    def _verify_requirement_linkage(
        *,
        requirement: AgentGovernanceHumanApprovalRequirement,
        request: ToolAuthorizationRequest,
        logical_invocation_fingerprint: LogicalInvocationFingerprint,
        pause_generation: int,
    ) -> None:
        if requirement.pause_generation != pause_generation:
            raise AgentGovernanceGrantVerificationError(
                "requirement generation mismatch"
            )
        if (
            requirement.logical_invocation_fingerprint.digest
            != logical_invocation_fingerprint.digest
        ):
            raise AgentGovernanceGrantVerificationError("fingerprint mismatch")
        auth = requirement.authorization_request
        if (
            auth.agent.agent_id != request.agent.agent_id
            or auth.tool_id != request.tool_id
            or auth.capability != request.capability
        ):
            raise AgentGovernanceGrantVerificationError("authorization request drift")

    @staticmethod
    def _verify_pending_grant_linkage(
        *,
        grant: AgentGovernanceHumanApprovalGrant,
        pending: AgentGovernanceHumanApprovalPending,
        requirement: AgentGovernanceHumanApprovalRequirement,
        logical_invocation_fingerprint: LogicalInvocationFingerprint,
        pause_generation: int,
    ) -> None:
        if pending.generation != pause_generation:
            raise AgentGovernanceGrantVerificationError("pending generation mismatch")
        if (
            grant.agent_governance_invocation_scope_id
            != pending.agent_governance_invocation_scope_id
        ):
            raise AgentGovernanceGrantVerificationError("agr scope mismatch")
        if grant.pending_generation != pending.generation:
            raise AgentGovernanceGrantVerificationError("pending generation mismatch")
        if grant.human_request_id != pending.human_request_id:
            raise AgentGovernanceGrantVerificationError("human request mismatch")
        if grant.pause_id != pending.pause_id:
            raise AgentGovernanceGrantVerificationError("pause mismatch")
        if grant.task_id != pending.task_id or grant.run_id != pending.run_id:
            raise AgentGovernanceGrantVerificationError("four-id mismatch")
        if (
            grant.attempt_id != pending.attempt_id
            or grant.execution_id != pending.execution_id
        ):
            raise AgentGovernanceGrantVerificationError("execution identity mismatch")
        if grant.task_id != requirement.task_id or grant.run_id != requirement.run_id:
            raise AgentGovernanceGrantVerificationError("requirement four-id mismatch")
        if (
            grant.attempt_id != requirement.attempt_id
            or grant.execution_id != requirement.execution_id
        ):
            raise AgentGovernanceGrantVerificationError(
                "requirement execution identity mismatch"
            )
        if grant.tool_id != pending.tool_id or grant.step_id != pending.step_id:
            raise AgentGovernanceGrantVerificationError("tool linkage mismatch")
        if grant.tool_id != requirement.tool_id or grant.step_id != requirement.step_id:
            raise AgentGovernanceGrantVerificationError(
                "requirement tool linkage mismatch"
            )
        if grant.agent_id != pending.agent_id or grant.tenant_id != pending.tenant_id:
            raise AgentGovernanceGrantVerificationError("agent linkage mismatch")
        if (
            grant.agent_id != requirement.agent_id
            or grant.tenant_id != requirement.tenant_id
        ):
            raise AgentGovernanceGrantVerificationError(
                "requirement agent linkage mismatch"
            )
        if grant.idempotency_key != pending.idempotency_key:
            raise AgentGovernanceGrantVerificationError("idempotency mismatch")
        if grant.idempotency_key != requirement.idempotency_key:
            raise AgentGovernanceGrantVerificationError(
                "requirement idempotency mismatch"
            )
        if grant.policy_provenance_digest != pending.policy_provenance_digest:
            raise AgentGovernanceGrantVerificationError("policy digest mismatch")
        if grant.policy_provenance_digest != requirement.policy_provenance_digest:
            raise AgentGovernanceGrantVerificationError(
                "requirement policy digest mismatch"
            )
        if (
            grant.logical_invocation_fingerprint.digest
            != logical_invocation_fingerprint.digest
        ):
            raise AgentGovernanceGrantVerificationError("fingerprint mismatch")
        if (
            grant.agent_governance_invocation_scope_id
            != requirement.agent_governance_invocation_scope_id
        ):
            raise AgentGovernanceGrantVerificationError("requirement scope mismatch")

    @staticmethod
    def _verify_applied_recovery_linkage(
        *,
        grant: AgentGovernanceHumanApprovalGrant,
        requirement: AgentGovernanceHumanApprovalRequirement,
        logical_invocation_fingerprint: LogicalInvocationFingerprint,
        pause_generation: int,
    ) -> None:
        if grant.pending_generation != pause_generation:
            raise AgentGovernanceGrantVerificationError("pending generation mismatch")
        if (
            grant.logical_invocation_fingerprint.digest
            != logical_invocation_fingerprint.digest
        ):
            raise AgentGovernanceGrantVerificationError("fingerprint mismatch")
        if (
            grant.agent_governance_invocation_scope_id
            != requirement.agent_governance_invocation_scope_id
        ):
            raise AgentGovernanceGrantVerificationError("agr scope mismatch")
        if grant.task_id != requirement.task_id or grant.run_id != requirement.run_id:
            raise AgentGovernanceGrantVerificationError("four-id mismatch")
        if (
            grant.attempt_id != requirement.attempt_id
            or grant.execution_id != requirement.execution_id
        ):
            raise AgentGovernanceGrantVerificationError("execution identity mismatch")
        if grant.tool_id != requirement.tool_id or grant.step_id != requirement.step_id:
            raise AgentGovernanceGrantVerificationError("tool linkage mismatch")
        if (
            grant.agent_id != requirement.agent_id
            or grant.tenant_id != requirement.tenant_id
        ):
            raise AgentGovernanceGrantVerificationError("agent linkage mismatch")
        if grant.idempotency_key != requirement.idempotency_key:
            raise AgentGovernanceGrantVerificationError("idempotency mismatch")
        if grant.policy_provenance_digest != requirement.policy_provenance_digest:
            raise AgentGovernanceGrantVerificationError("policy digest mismatch")

    def reject_foreign_approval_artifact(self, artifact: object) -> None:
        if isinstance(artifact, DeclarativeHitlApprovalGrant):
            raise AgentGovernanceGrantVerificationError("declarative grant rejected")
        if isinstance(artifact, GovernedContinuationApprovalGrant):
            raise AgentGovernanceGrantVerificationError("mse grant rejected")
        if isinstance(artifact, str):
            raise AgentGovernanceGrantVerificationError("raw string grant rejected")
        raise AgentGovernanceGrantVerificationError("unsupported approval artifact")


__all__ = [
    "AgentGovernanceGrantVerificationError",
    "AgentGovernanceGrantVerifier",
]
