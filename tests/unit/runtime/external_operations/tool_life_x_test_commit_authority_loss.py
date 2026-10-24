# © Artur Czarnecki. All rights reserved.

"""Test helpers for durable commit-authority-loss decisions (L3-R8/R9/R10)."""

from __future__ import annotations

from intergrax.contracts.certified_owner_loss_evidence import CertifiedOwnerLossReason
from intergrax.runtime.external_operations.sqlite_tool_life_durable_stores import (
    SqliteToolLifeDurableStoresBundle,
)
from intergrax.runtime.external_operations.worker_execution_commit_authority_loss import (
    WorkerExecutionCommitAuthorityLossDecisionAuthority,
)


def issue_sqlite_commit_authority_loss_decision_for_bound_operation(
    bundle: SqliteToolLifeDurableStoresBundle,
    *,
    tenant_id: str,
    operation_id: str,
    loss_reason: CertifiedOwnerLossReason = CertifiedOwnerLossReason.WORKER_LOST,
) -> None:
    authority = WorkerExecutionCommitAuthorityLossDecisionAuthority(
        authoritative_state_store=bundle.external_operation_state,
        decision_port=bundle.worker_execution_ownership_revocation,
    )
    authority.issue_for_bound_operation(
        tenant_id=tenant_id,
        operation_id=operation_id,
        loss_reason=loss_reason,
    )
