# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from typing import Dict, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel

from intergrax.contracts.idempotency_store import (
    ClaimOutcome,
    ClaimResult,
    IdempotencyStore,
    InvocationClaim,
    InvocationStatus,
)
from intergrax.contracts.lease_claim import StaleClaimError
from intergrax.contracts.persistence_topology import PersistenceTopology
from intergrax.tools.execution_models import ToolExecutionResult


class _LedgerEntry:
    __slots__ = ("status", "result", "claim")

    def __init__(
        self,
        status: InvocationStatus,
        result: Optional[ToolExecutionResult[BaseModel]],
        claim: InvocationClaim | None,
    ) -> None:
        self.status = status
        self.result = result
        self.claim = claim


class InMemoryIdempotencyStore(IdempotencyStore):
    """Process-local ledger — state is lost on process restart."""

    @property
    def persistence_topology(self) -> PersistenceTopology:
        return PersistenceTopology.PROCESS_LOCAL

    def __init__(self) -> None:
        self._store: Dict[Tuple[str, str], _LedgerEntry] = {}
        self._lock = threading.Lock()

    def get_status(
        self,
        tenant_id: str,
        key: str,
    ) -> Optional[InvocationStatus]:
        with self._lock:
            entry = self._store.get((tenant_id, key))
            if entry is None:
                return None
            return entry.status

    def claim(
        self,
        tenant_id: str,
        key: str,
        owner_id: str,
        lease_seconds: int,
    ) -> ClaimResult:
        now = datetime.now(UTC)
        lease_expires_at = now + timedelta(seconds=lease_seconds)
        composite_key = (tenant_id, key)

        with self._lock:
            entry = self._store.get(composite_key)
            if entry is None:
                claim = InvocationClaim(
                    tenant_id=tenant_id,
                    key=key,
                    owner_id=owner_id,
                    lease_expires_at=lease_expires_at,
                    fence=1,
                )
                self._store[composite_key] = _LedgerEntry(
                    InvocationStatus.STARTED,
                    None,
                    claim,
                )
                return ClaimResult(outcome=ClaimOutcome.ACQUIRED, claim=claim)

            if entry.status == InvocationStatus.COMPLETED:
                return ClaimResult(
                    outcome=ClaimOutcome.REPLAY_COMPLETED,
                    completed_result=entry.result,
                )

            if entry.status == InvocationStatus.UNCERTAIN:
                return ClaimResult(outcome=ClaimOutcome.UNCERTAIN)

            assert entry.claim is not None
            if entry.claim.lease_expires_at > now:
                if entry.claim.owner_id == owner_id:
                    return ClaimResult(outcome=ClaimOutcome.ACQUIRED, claim=entry.claim)
                return ClaimResult(outcome=ClaimOutcome.BLOCKED_ACTIVE)

            self._store[composite_key] = _LedgerEntry(
                InvocationStatus.UNCERTAIN,
                None,
                entry.claim,
            )
            return ClaimResult(outcome=ClaimOutcome.UNCERTAIN)

    def complete_with_claim(
        self,
        tenant_id: str,
        key: str,
        claim: InvocationClaim,
        result: ToolExecutionResult[BaseModel],
        completed_ttl_seconds: Optional[int] = None,
    ) -> None:
        del completed_ttl_seconds
        composite_key = (tenant_id, key)
        with self._lock:
            entry = self._store.get(composite_key)
            if entry is None or entry.status != InvocationStatus.STARTED or entry.claim is None:
                raise StaleClaimError(
                    f"Cannot complete key={key}: missing or invalid active claim.",
                )
            current = entry.claim
            if current.owner_id != claim.owner_id or current.fence != claim.fence:
                raise StaleClaimError(
                    f"Stale completion rejected for key={key} fence={claim.fence}.",
                )
            self._store[composite_key] = _LedgerEntry(
                InvocationStatus.COMPLETED,
                result,
                current,
            )

    def record_started(
        self,
        tenant_id: str,
        key: str,
        lease_seconds: Optional[int] = None,
    ) -> None:
        lease = lease_seconds if lease_seconds is not None else 300
        owner_id = f"legacy-{uuid4().hex}"
        result = self.claim(tenant_id, key, owner_id, lease)
        if result.outcome == ClaimOutcome.REPLAY_COMPLETED:
            raise RuntimeError(f"Invocation already completed for key={key}")
        if result.outcome == ClaimOutcome.BLOCKED_ACTIVE:
            raise RuntimeError(f"Invocation already started for key={key}")
        if result.outcome == ClaimOutcome.UNCERTAIN:
            raise RuntimeError(
                f"Invocation uncertain for key={key}; reconciliation required.",
            )

    def record_completed(
        self,
        tenant_id: str,
        key: str,
        result: ToolExecutionResult[BaseModel],
        completed_ttl_seconds: Optional[int] = None,
    ) -> None:
        del completed_ttl_seconds
        composite_key = (tenant_id, key)
        with self._lock:
            entry = self._store.get(composite_key)
            if entry is None or entry.status != InvocationStatus.STARTED or entry.claim is None:
                raise RuntimeError("Cannot mark completed without STARTED state.")
            self._store[composite_key] = _LedgerEntry(
                InvocationStatus.COMPLETED,
                result,
                entry.claim,
            )

    def get_completed_result(
        self,
        tenant_id: str,
        key: str,
    ) -> Optional[ToolExecutionResult[BaseModel]]:
        with self._lock:
            entry = self._store.get((tenant_id, key))
            if entry is None or entry.status != InvocationStatus.COMPLETED:
                return None
            return entry.result
