# © Artur Czarnecki. All rights reserved.

"""Real Unified Execution interop proof for execution linkage (COLLAB-WORK-2F)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.collaborative_work.execution_link_service import CollaborativeWorkExecutionLinkService
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryWorkItemExecutionLinkRepository,
    InMemoryWorkItemRepository,
)
from intergrax.collaborative_work.repository import CreateWorkItemCommand
from intergrax.contracts.collaborative_work import LinkWorkItemExecutionRequest, WorkItemState
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
)
from intergrax.contracts.execution_provenance import ExecutionProvenanceRef
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.long_running.runtime_checkpoint import RuntimeCheckpoint
from intergrax.runtime.task.task import Task

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)


def test_real_task_and_runtime_context_provenance_persists_via_service() -> None:
    task = Task(tenant_id=_TENANT, user_id="user-1", message="interop")
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    runtime_context = RuntimeExecutionContext(
        task_id=task.task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        agent_id="agent-1",
    )
    provenance = ExecutionProvenanceRef(
        task_id=runtime_context.task_id,
        run_id=runtime_context.run_id,
        attempt_id=runtime_context.attempt_id,
        execution_id=runtime_context.execution_id,
    )

    work_item_repo = InMemoryWorkItemRepository()
    execution_link_repo = InMemoryWorkItemExecutionLinkRepository()
    work_item_repo.create(
        CreateWorkItemCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_item_id="work-item-interop",
            created_by_principal_id="principal-creator",
            state=WorkItemState.OPEN,
            created_at=_NOW,
            updated_at=_NOW,
        ),
    )
    service = CollaborativeWorkExecutionLinkService(
        work_item_repository=work_item_repo,
        execution_link_repository=execution_link_repo,
        clock=lambda: _NOW,
    )
    linked = service.link_execution(
        LinkWorkItemExecutionRequest(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_item_id="work-item-interop",
            execution=provenance,
            idempotency_key="interop-idem",
        ),
    )
    loaded = execution_link_repo.get(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        execution_link_id=linked.execution_link_id,
    )
    assert loaded == linked
    assert loaded is not None
    assert loaded.execution.task_id == task.task_id
    assert loaded.execution.run_id == run_id
    assert loaded.execution.attempt_id == attempt_id
    assert loaded.execution.execution_id == execution_id
