# © Artur Czarnecki. All rights reserved.

"""Persisted task-row fencing via existing ``LeaseOwnership`` fields (ARCH-FREEZE B)."""

from __future__ import annotations

from datetime import UTC, datetime

from intergrax.contracts.idempotency_store import InvocationClaim
from intergrax.contracts.lease_claim import LeaseOwnership

_ROW_OWNER = "claim_owner_id"
_ROW_LEASE = "claim_lease_expires_at"
_ROW_FENCE = "claim_fence"
_ROW_LEDGER_KEY = "claim_ledger_key"


def task_row_ledger_key(*, logical_task_name: str, idempotency_key: str | None, task_id: str) -> str:
    if idempotency_key:
        return f"{logical_task_name}:{idempotency_key}"
    return f"document_store_task:{task_id}"


def write_claim_to_row(row: dict[str, object], *, claim: InvocationClaim, ledger_key: str) -> None:
    row[_ROW_OWNER] = claim.owner_id
    row[_ROW_LEASE] = claim.lease_expires_at.astimezone(UTC).isoformat()
    row[_ROW_FENCE] = claim.fence
    row[_ROW_LEDGER_KEY] = ledger_key


def read_claim_from_row(row: dict[str, object]) -> tuple[LeaseOwnership, str] | None:
    owner = row.get(_ROW_OWNER)
    lease_raw = row.get(_ROW_LEASE)
    fence = row.get(_ROW_FENCE)
    ledger_key = row.get(_ROW_LEDGER_KEY)
    if not isinstance(owner, str) or not owner.strip():
        return None
    if not isinstance(lease_raw, str) or not lease_raw.strip():
        return None
    if not isinstance(fence, int):
        return None
    if not isinstance(ledger_key, str) or not ledger_key.strip():
        return None
    lease_expires_at = datetime.fromisoformat(lease_raw)
    if lease_expires_at.tzinfo is None:
        lease_expires_at = lease_expires_at.replace(tzinfo=UTC)
    return (
        LeaseOwnership(
            owner_id=owner,
            lease_expires_at=lease_expires_at,
            fence=fence,
        ),
        ledger_key,
    )


def row_claim_is_live(row: dict[str, object], *, now: datetime | None = None) -> bool:
    parsed = read_claim_from_row(row)
    if parsed is None:
        return False
    ownership, _ = parsed
    moment = now or datetime.now(UTC)
    return ownership.lease_expires_at.astimezone(UTC) > moment.astimezone(UTC)


def invocation_claim_from_row(
    row: dict[str, object],
    *,
    tenant_id: str,
) -> InvocationClaim | None:
    parsed = read_claim_from_row(row)
    if parsed is None:
        return None
    ownership, ledger_key = parsed
    return InvocationClaim(
        owner_id=ownership.owner_id,
        lease_expires_at=ownership.lease_expires_at,
        fence=ownership.fence,
        tenant_id=tenant_id,
        key=ledger_key,
    )


def row_claim_matches(row: dict[str, object], claim: LeaseOwnership) -> bool:
    parsed = read_claim_from_row(row)
    if parsed is None:
        return False
    ownership, _ = parsed
    return (
        ownership.owner_id == claim.owner_id
        and ownership.fence == claim.fence
        and ownership.lease_expires_at == claim.lease_expires_at
    )


__all__ = [
    "invocation_claim_from_row",
    "read_claim_from_row",
    "row_claim_is_live",
    "row_claim_matches",
    "task_row_ledger_key",
    "write_claim_to_row",
]
