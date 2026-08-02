# © Artur Czarnecki. All rights reserved.

"""Startup recovery for interrupted connected-source synchronization operations."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.tools.registry.wiring import ToolWiringContext
from local_workspace_application.workspaces.connected_source_source_projection import (
    repair_connected_source_source_projections_for_tenant,
)
from local_workspace_application.workspaces.connected_source_sync_enqueue import (
    repair_connected_source_pending_enqueue,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository


@dataclass(frozen=True, slots=True)
class ConnectedSourceRecoveryResult:
    operations_seen: int = 0
    operations_requeued: int = 0
    errors: int = 0
    sources_repaired: int = 0


class ConnectedSourceRecoveryService:
    def __init__(
        self,
        repository: ManagedWorkspaceRepository,
        wiring_context: ToolWiringContext,
        tenant_ids: tuple[str, ...] = (),
    ) -> None:
        self._repository = repository
        self._wiring_context = wiring_context
        self._tenant_ids = tenant_ids

    def recover_running_operations(self) -> ConnectedSourceRecoveryResult:
        sources_repaired = 0
        for tenant_id in self._tenant_ids:
            if not tenant_id.strip():
                continue
            sources_repaired += repair_connected_source_source_projections_for_tenant(
                repository=self._repository,
                tenant_id=tenant_id,
            )
        seen, repaired, errors = repair_connected_source_pending_enqueue(
            repository=self._repository,
            wiring_context=self._wiring_context,
            tenant_ids=self._tenant_ids,
        )
        return ConnectedSourceRecoveryResult(
            operations_seen=seen,
            operations_requeued=repaired,
            errors=errors,
            sources_repaired=sources_repaired,
        )
