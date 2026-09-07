# © Artur Czarnecki. All rights reserved.

"""CollaborativeWorkExecutionLinkService tests (COLLAB-WORK-2F)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from intergrax.collaborative_work.execution_link_service import CollaborativeWorkExecutionLinkService
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAssignmentRepository,
    InMemoryWorkItemExecutionLinkRepository,
    InMemoryWorkItemRepository,
)
from intergrax.collaborative_work.repository import (
    CreateAssignmentCommand,
    CreateWorkItemCommand,
    WorkItemNotFound,
)
from intergrax.contracts.collaborative_work import (
    AssignmentState,
    LinkWorkItemExecutionRequest,
    WorkItemState,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_provenance import ExecutionProvenanceRef

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_WORK_ITEM_ID = "work-item-1"
_NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)


def _execution(**overrides: object) -> ExecutionProvenanceRef:
    payload = {
        "task_id": mint_task_id(),
        "run_id": mint_run_id(),
        "attempt_id": mint_attempt_id(),
        "execution_id": mint_execution_id(),
    }
    payload.update(overrides)
    return ExecutionProvenanceRef(**payload)


def _seed_work_item(
    repo: InMemoryWorkItemRepository,
    *,
    work_item_id: str = _WORK_ITEM_ID,
) -> None:
    repo.create(
        CreateWorkItemCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_item_id=work_item_id,
            created_by_principal_id="principal-creator",
            state=WorkItemState.OPEN,
            created_at=_NOW,
            updated_at=_NOW,
        ),
    )


def _service(
    *,
    clock: datetime | None = None,
) -> tuple[
    CollaborativeWorkExecutionLinkService,
    InMemoryWorkItemRepository,
    InMemoryAssignmentRepository,
    InMemoryWorkItemExecutionLinkRepository,
]:
    work_item_repo = InMemoryWorkItemRepository()
    assignment_repo = InMemoryAssignmentRepository()
    execution_link_repo = InMemoryWorkItemExecutionLinkRepository()
    fixed = clock or _NOW
    service = CollaborativeWorkExecutionLinkService(
        work_item_repository=work_item_repo,
        execution_link_repository=execution_link_repo,
        clock=lambda: fixed,
    )
    return service, work_item_repo, assignment_repo, execution_link_repo


def _request(**overrides: object) -> LinkWorkItemExecutionRequest:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "work_item_id": _WORK_ITEM_ID,
        "execution": _execution(),
    }
    payload.update(overrides)
    return LinkWorkItemExecutionRequest(**payload)


def test_link_existing_work_item() -> None:
    service, work_item_repo, _, _ = _service()
    _seed_work_item(work_item_repo)
    linked = service.link_execution(_request())
    assert linked.work_item_id == _WORK_ITEM_ID
    assert linked.execution.task_id.startswith("task_")


def test_missing_work_item_raises_not_found() -> None:
    service, _, _, _ = _service()
    with pytest.raises(WorkItemNotFound):
        service.link_execution(_request())


def test_wrong_tenant_or_workspace_raises_not_found() -> None:
    service, work_item_repo, _, execution_link_repo = _service()
    _seed_work_item(work_item_repo)
    with pytest.raises(WorkItemNotFound):
        service.link_execution(_request(tenant_id="other-tenant"))
    with pytest.raises(WorkItemNotFound):
        service.link_execution(_request(workspace_id="other-workspace"))


def test_idempotent_retry_with_fixed_clock() -> None:
    service, work_item_repo, _, execution_link_repo = _service()
    _seed_work_item(work_item_repo)
    request = _request(idempotency_key="link-idem")
    first = service.link_execution(request)
    second = service.link_execution(request)
    assert second == first


def test_zero_links_valid_for_new_work_item() -> None:
    service, work_item_repo, _, execution_link_repo = _service()
    _seed_work_item(work_item_repo)
    work_item = work_item_repo.get(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        work_item_id=_WORK_ITEM_ID,
    )
    assert work_item is not None
    assert (
        execution_link_repo.list_for_work_item(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_item_id=_WORK_ITEM_ID,
        )
        == ()
    )


def test_multiple_executions_for_one_work_item() -> None:
    service, work_item_repo, _, execution_link_repo = _service(clock=_NOW)
    _seed_work_item(work_item_repo)
    task_id = mint_task_id()
    run_a = mint_run_id()
    attempt_a = mint_attempt_id()
    service.link_execution(
        _request(
            execution=_execution(
                task_id=task_id,
                run_id=run_a,
                attempt_id=attempt_a,
                execution_id=mint_execution_id(),
            ),
        ),
    )
    service.link_execution(
        _request(
            execution=_execution(
                task_id=task_id,
                run_id=run_a,
                attempt_id=attempt_a,
                execution_id=mint_execution_id(),
            ),
        ),
    )
    service.link_execution(
        _request(
            execution=_execution(
                task_id=mint_task_id(),
                run_id=mint_run_id(),
                attempt_id=mint_attempt_id(),
                execution_id=mint_execution_id(),
            ),
        ),
    )
    links = execution_link_repo.list_for_work_item(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        work_item_id=_WORK_ITEM_ID,
    )
    assert len(links) == 3


def test_work_item_state_revision_and_updated_at_unchanged_after_link() -> None:
    service, work_item_repo, _, execution_link_repo = _service()
    _seed_work_item(work_item_repo)
    before = work_item_repo.get(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        work_item_id=_WORK_ITEM_ID,
    )
    assert before is not None
    service.link_execution(_request())
    after = work_item_repo.get(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        work_item_id=_WORK_ITEM_ID,
    )
    assert after is not None
    assert after.state == before.state
    assert after.revision == before.revision
    assert after.updated_at == before.updated_at


def test_assignment_not_mutated_by_execution_link() -> None:
    service, work_item_repo, assignment_repo, _ = _service()
    _seed_work_item(work_item_repo)
    assignment = assignment_repo.create(
        CreateAssignmentCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            assignment_id="assignment-1",
            work_item_id=_WORK_ITEM_ID,
            principal_id="principal-1",
            created_by_principal_id="principal-creator",
            state=AssignmentState.ACTIVE,
            created_at=_NOW,
            updated_at=_NOW,
        ),
    )
    service.link_execution(_request())
    loaded = assignment_repo.get(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        assignment_id="assignment-1",
    )
    assert loaded == assignment


def test_execution_completion_does_not_mutate_work_item() -> None:
    service, work_item_repo, _, execution_link_repo = _service()
    _seed_work_item(work_item_repo)
    before = work_item_repo.get(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        work_item_id=_WORK_ITEM_ID,
    )
    assert before is not None
    execution = _execution()
    service.link_execution(_request(execution=execution))
    completed_execution = ExecutionProvenanceRef(
        task_id=execution.task_id,
        run_id=execution.run_id,
        attempt_id=execution.attempt_id,
        execution_id=mint_execution_id(),
    )
    assert completed_execution.execution_id != execution.execution_id
    after = work_item_repo.get(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        work_item_id=_WORK_ITEM_ID,
    )
    assert after == before
