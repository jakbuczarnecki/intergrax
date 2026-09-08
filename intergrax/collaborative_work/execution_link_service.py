# © Artur Czarnecki. All rights reserved.

"""Trusted WorkItem ↔ Unified Execution provenance association service (COLLAB-WORK-2F)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from intergrax.collaborative_work.repository import (
    CreateWorkItemExecutionLinkCommand,
    WorkItemExecutionLinkRepository,
    WorkItemNotFound,
    WorkItemRepository,
)
from intergrax.contracts.collaborative_work import (
    LinkWorkItemExecutionRequest,
    WorkItemExecutionLink,
    mint_execution_link_id,
)


class CollaborativeWorkExecutionLinkService:
    """Domain boundary for append-only WorkItem execution provenance linkage."""

    def __init__(
        self,
        *,
        work_item_repository: WorkItemRepository,
        execution_link_repository: WorkItemExecutionLinkRepository,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._work_item_repository = work_item_repository
        self._execution_link_repository = execution_link_repository
        self._clock = clock or (lambda: datetime.now(UTC))

    def link_execution(self, request: LinkWorkItemExecutionRequest) -> WorkItemExecutionLink:
        work_item = self._work_item_repository.get(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            work_item_id=request.work_item_id,
        )
        if work_item is None:
            raise WorkItemNotFound("work item was not found")

        execution_link_id = mint_execution_link_id(idempotency_key=request.idempotency_key)
        linked_at = self._clock()
        return self._execution_link_repository.create(
            CreateWorkItemExecutionLinkCommand(
                tenant_id=request.tenant_id,
                workspace_id=request.workspace_id,
                execution_link_id=execution_link_id,
                work_item_id=request.work_item_id,
                execution=request.execution,
                linked_at=linked_at,
                idempotency_key=request.idempotency_key,
            ),
        )
