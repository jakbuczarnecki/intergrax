# © Artur Czarnecki. All rights reserved.

"""Bridge ``ReliabilityProfile.idempotency_store`` with ACP side-effect ledger (ACP-CLOSE-PROD-6)."""

from __future__ import annotations

from pydantic import BaseModel

from intergrax.agents.persistence.side_effect_ledger import SideEffectLedger
from intergrax.contracts.idempotency_store import IdempotencyStore


class SideEffectCommitPayload(BaseModel):
    """Minimal durable payload for cross-run declarative side-effect dedupe."""

    tool_id: str
    external_ref: str | None = None


def should_skip_side_effect_replay(
    *,
    idempotency_key: str,
    ledger: SideEffectLedger | None = None,
    idempotency_store: IdempotencyStore | None = None,
    tenant_id: str = "default",
) -> bool:
    """True when the key is committed in-run (ledger) or cross-run (idempotency store)."""
    if ledger is not None and ledger.should_skip_replay(idempotency_key):
        return True
    if idempotency_store is None:
        return False
    completed = idempotency_store.get_completed_result(tenant_id, idempotency_key)
    return completed is not None and completed.success


def resolve_external_ref_from_store(
    *,
    idempotency_store: IdempotencyStore | None,
    tenant_id: str,
    idempotency_key: str,
) -> str | None:
    if idempotency_store is None:
        return None
    completed = idempotency_store.get_completed_result(tenant_id, idempotency_key)
    if completed is None or not completed.success or completed.output is None:
        return None
    output = completed.output
    if isinstance(output, SideEffectCommitPayload):
        return output.external_ref
    external_ref = output.model_dump().get("external_ref")
    return external_ref if isinstance(external_ref, str) else None
