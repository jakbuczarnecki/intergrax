# © Artur Czarnecki. All rights reserved.

"""Pluginable Agent Governance grant lifecycle port (semantics only, UCA-6C-R6-R5)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from intergrax.contracts.agent_governance_hitl import (
    AgentGovernanceGrantLifecycleRecord,
    AgentGovernanceHumanApprovalGrant,
    LogicalInvocationFingerprint,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
)


class AgentGovernanceGrantLifecycleOutcome(StrEnum):
    APPLIED = "applied"
    STALE_REVISION = "stale_revision"
    CONFLICT = "conflict"
    NOT_FOUND = "not_found"
    INVALID_STATE = "invalid_state"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class AgentGovernanceGrantLifecycleMutationResult:
    outcome: AgentGovernanceGrantLifecycleOutcome
    record: AgentGovernanceGrantLifecycleRecord | None = None


class AgentGovernanceGrantLifecyclePort(ABC):
    """Mutates canonical Task governance grant lifecycle — not a second store."""

    @abstractmethod
    def load(
        self,
        *,
        task_id: TaskId,
        tenant_id: str,
    ) -> AgentGovernanceGrantLifecycleRecord | None:
        ...

    @abstractmethod
    def reserve(
        self,
        *,
        task_id: TaskId,
        tenant_id: str,
        expected_lifecycle_revision: int,
        grant_id: str,
        logical_invocation_fingerprint: LogicalInvocationFingerprint,
        pause_generation: int,
        agent_governance_invocation_scope_id: str,
        task_id_link: TaskId,
        run_id: RunId,
        attempt_id: AttemptId,
        execution_id: ExecutionId,
        policy_provenance_digest: str | None,
        owner_id: str,
        lease_expires_at: datetime,
    ) -> AgentGovernanceGrantLifecycleMutationResult:
        """CAS AVAILABLE → RESERVED with durable linkage."""

    @abstractmethod
    def reclaim_reservation(
        self,
        *,
        task_id: TaskId,
        tenant_id: str,
        expected_lifecycle_revision: int,
        owner_id: str,
        lease_expires_at: datetime,
        expected_fence: int,
    ) -> AgentGovernanceGrantLifecycleMutationResult:
        """Reclaim expired RESERVED lease (monotonic fence)."""

    @abstractmethod
    def mark_applied(
        self,
        *,
        task_id: TaskId,
        tenant_id: str,
        expected_lifecycle_revision: int,
        owner_id: str,
        fence: int,
    ) -> AgentGovernanceGrantLifecycleMutationResult:
        """CAS RESERVED → APPLIED after fresh governance ALLOW."""

    @abstractmethod
    def terminalize(
        self,
        *,
        task_id: TaskId,
        tenant_id: str,
        expected_lifecycle_revision: int,
        reason: str,
    ) -> AgentGovernanceGrantLifecycleMutationResult:
        ...

    @abstractmethod
    def persist_available_grant(
        self,
        *,
        task_id: TaskId,
        tenant_id: str,
        expected_lifecycle_revision: int,
        grant: AgentGovernanceHumanApprovalGrant,
    ) -> AgentGovernanceGrantLifecycleMutationResult:
        """Persist human APPROVE result as AVAILABLE grant."""


__all__ = [
    "AgentGovernanceGrantLifecycleMutationResult",
    "AgentGovernanceGrantLifecycleOutcome",
    "AgentGovernanceGrantLifecyclePort",
]
