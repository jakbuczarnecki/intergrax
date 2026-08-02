# © Artur Czarnecki. All rights reserved.

"""Checkpoint-authoritative restart decisions for connected-source sync."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.vendor_knowledge.sync_contracts import KnowledgeSyncCheckpointRepository
from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceReconciliationStateV1,
)
from local_workspace_application.workspaces.models import WorkspaceOperation
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository


@dataclass(frozen=True, slots=True)
class ConnectedSourceRestartDecision:
    restart: bool
    operation: WorkspaceOperation
    checkpoint_binding_version: int | None = None


def resolve_connected_source_restart(
    *,
    repository: ManagedWorkspaceRepository,
    checkpoint_repository: KnowledgeSyncCheckpointRepository,
    tenant_id: str,
    binding_ref: str,
    binding_configuration_version: int,
    operation: WorkspaceOperation,
) -> ConnectedSourceRestartDecision:
    checkpoint = checkpoint_repository.get(tenant_id=tenant_id, binding_id=binding_ref)
    valid_checkpoint = (
        checkpoint is not None
        and checkpoint.binding_configuration_version == binding_configuration_version
    )

    reconciliation_state = operation.connected_source_reconciliation_state
    if reconciliation_state is None:
        reconciliation_state = ConnectedSourceReconciliationStateV1.NEW_RECONCILIATION

    if valid_checkpoint:
        if reconciliation_state is not ConnectedSourceReconciliationStateV1.CONTINUATION:
            operation = operation.model_copy(
                update={
                    "connected_source_reconciliation_state": (
                        ConnectedSourceReconciliationStateV1.CONTINUATION
                    )
                }
            )
            repository.put_operation(operation)
        return ConnectedSourceRestartDecision(
            restart=False,
            operation=operation,
            checkpoint_binding_version=checkpoint.binding_configuration_version,
        )

    if reconciliation_state is ConnectedSourceReconciliationStateV1.CONTINUATION:
        operation = operation.model_copy(
            update={
                "connected_source_reconciliation_state": (
                    ConnectedSourceReconciliationStateV1.NEW_RECONCILIATION
                )
            }
        )
        repository.put_operation(operation)

    return ConnectedSourceRestartDecision(
        restart=True,
        operation=operation,
        checkpoint_binding_version=None,
    )
