# © Artur Czarnecki. All rights reserved.

"""Test helpers for durable commit-authority-loss decisions (L3-R8/R9/R10)."""

from __future__ import annotations

from intergrax.contracts.certified_owner_loss_evidence import CertifiedOwnerLossReason
from intergrax.runtime.external_operations.certified_owner_loss_composition import (
    require_certified_owner_loss_composition,
)
from intergrax.runtime.external_operations.sqlite_tool_life_durable_stores import (
    SqliteToolLifeDurableStoresBundle,
)


def record_sqlite_revocation_for_bound_operation(
    bundle: SqliteToolLifeDurableStoresBundle,
    *,
    tenant_id: str,
    operation_id: str,
    loss_reason: CertifiedOwnerLossReason = CertifiedOwnerLossReason.WORKER_LOST,
) -> None:
    """Record authoritative revocation; decision evidence is minted in the same SQLite TX."""
    recorder = require_certified_owner_loss_composition().worker_execution_revocation_recorder
    if recorder is None:
        raise RuntimeError("revocation recorder not configured for qualification composition")
    recorder.record_revocation_for_bound_operation(
        tenant_id=tenant_id,
        operation_id=operation_id,
        loss_reason=loss_reason,
    )


def issue_sqlite_commit_authority_loss_decision_for_bound_operation(
    bundle: SqliteToolLifeDurableStoresBundle,
    *,
    tenant_id: str,
    operation_id: str,
    loss_reason: CertifiedOwnerLossReason = CertifiedOwnerLossReason.WORKER_LOST,
) -> None:
    """Backward-compatible alias — decisions are no longer issued outside revocation."""
    record_sqlite_revocation_for_bound_operation(
        bundle,
        tenant_id=tenant_id,
        operation_id=operation_id,
        loss_reason=loss_reason,
    )
