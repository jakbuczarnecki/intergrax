# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict

from intergrax.contracts.lease_claim import LeaseOwnership
from intergrax.contracts.persistence_topology import PersistenceTopology
from intergrax.contracts.validation import validate_content_digest
from intergrax.tools.execution_models import ToolExecutionResult


class InvocationStatus(str, Enum):
    """
    Persistent invocation state.

    STARTED   — active claim; external effect may be in flight.
    COMPLETED — execution finished and result is stored.
    UNCERTAIN — lease expired without completion; outcome cannot be proven.
    """

    STARTED = "started"
    COMPLETED = "completed"
    UNCERTAIN = "uncertain"


class ClaimOutcome(str, Enum):
    """Result of an atomic invocation claim attempt."""

    ACQUIRED = "acquired"
    REPLAY_COMPLETED = "replay_completed"
    BLOCKED_ACTIVE = "blocked_active"
    UNCERTAIN = "uncertain"


class InvocationOperationIdentity(BaseModel):
    """Logical operation bound to one idempotency ledger entry (TOOLS-04)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_id: str
    operation_fingerprint: str

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        if not self.tool_id:
            raise ValueError("tool_id is required for InvocationOperationIdentity")
        validate_content_digest(self.operation_fingerprint)


class InvocationClaim(LeaseOwnership):
    """Active or historical ownership record for one idempotency key."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    key: str
    operation_identity: InvocationOperationIdentity | None = None


class ClaimResult(BaseModel):
    """Outcome of ``claim`` including optional ownership or cached replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: ClaimOutcome
    claim: InvocationClaim | None = None
    completed_result: ToolExecutionResult[BaseModel] | None = None


class InvocationUncertaintyError(RuntimeError):
    """External side-effect outcome cannot be determined; reconciliation required."""


class IdempotencyOperationConflictError(RuntimeError):
    """Same tenant/key reused for a different logical operation; replay forbidden."""


class ActiveInvocationClaimError(RuntimeError):
    """Another owner holds a valid active claim."""


def assert_operation_identity_compatible(
    stored: InvocationOperationIdentity | None,
    requested: InvocationOperationIdentity | None,
) -> None:
    """Fail closed when ledger operation identity does not match the request."""
    if stored is None and requested is None:
        return
    if stored is None or requested is None or stored != requested:
        raise IdempotencyOperationConflictError(
            "Idempotency key is bound to a different logical operation.",
        )


class IdempotencyStore(ABC):
    """
    Ledger-based idempotency port for tool invocations.

    Domain port — exposes ``persistence_topology`` for host deployment qualification.
    Provides duplicate suppression and execution-uncertainty tracking via atomic
    claim/owner/lease/fence semantics (PCM-SIDE-EFFECT-COORDINATION-INTEGRITY).
    """

    @property
    @abstractmethod
    def persistence_topology(self) -> PersistenceTopology:
        """Declared deployment topology this implementation can satisfy."""
        ...

    @abstractmethod
    def get_status(
        self,
        tenant_id: str,
        key: str,
    ) -> Optional[InvocationStatus]:
        """Returns current invocation status or None if not recorded."""
        ...

    @abstractmethod
    def claim(
        self,
        tenant_id: str,
        key: str,
        owner_id: str,
        lease_seconds: int,
        operation_identity: InvocationOperationIdentity | None = None,
    ) -> ClaimResult:
        """
        Atomically acquire invocation ownership or classify existing state.

        Only one active owner may succeed. Expired claims without completion
        transition to UNCERTAIN — they must not be treated as safe retry.

        When ``operation_identity`` is provided, replay requires an exact match;
        a different logical operation under the same key fails closed.
        """
        ...

    @abstractmethod
    def complete_with_claim(
        self,
        tenant_id: str,
        key: str,
        claim: InvocationClaim,
        result: ToolExecutionResult[BaseModel],
        completed_ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Transition STARTED -> COMPLETED when ``claim`` matches current ownership.

        Raises ``StaleClaimError`` when fence or owner is superseded.
        """
        ...

    @abstractmethod
    def mark_uncertain_with_claim(
        self,
        tenant_id: str,
        key: str,
        claim: InvocationClaim,
    ) -> None:
        """
        Transition STARTED -> UNCERTAIN when external outcome cannot be proven.

        Raises ``StaleClaimError`` when fence or owner is superseded.
        """
        ...

    @abstractmethod
    def record_started(
        self,
        tenant_id: str,
        key: str,
        lease_seconds: Optional[int] = None,
    ) -> None:
        """
        Legacy STARTED transition without typed ownership.

        Prefer ``claim`` for side-effect coordination. Implementations should
        remain atomic for NONE -> STARTED.
        """

    @abstractmethod
    def record_completed(
        self,
        tenant_id: str,
        key: str,
        result: ToolExecutionResult[BaseModel],
        completed_ttl_seconds: Optional[int] = None,
    ) -> None:
        """Legacy STARTED -> COMPLETED without fence validation."""

    @abstractmethod
    def get_completed_result(
        self,
        tenant_id: str,
        key: str,
    ) -> Optional[ToolExecutionResult[BaseModel]]:
        """Returns previously completed execution result if exists."""
        ...
