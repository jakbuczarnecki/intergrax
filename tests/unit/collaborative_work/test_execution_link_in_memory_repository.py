# © Artur Czarnecki. All rights reserved.

"""Repository contract tests for WorkItem execution links (COLLAB-WORK-2F)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from intergrax.collaborative_work.in_memory_repository import (
    InMemoryWorkItemExecutionLinkRepository,
)
from intergrax.collaborative_work.persistence import open_sqlite_collaborative_work_repositories
from intergrax.collaborative_work.repository import (
    CreateWorkItemExecutionLinkCommand,
    WorkItemExecutionLinkAlreadyExists,
    WorkItemExecutionLinkIdempotencyConflict,
    WorkItemExecutionLinkRepository,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_provenance import ExecutionProvenanceRef

pytestmark = pytest.mark.unit

_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"
_WORKSPACE_A = "workspace-a"
_WORKSPACE_B = "workspace-b"
_CREATED_AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
_LATER = _CREATED_AT + timedelta(minutes=5)


def _execution(**overrides: object) -> ExecutionProvenanceRef:
    payload = {
        "task_id": mint_task_id(),
        "run_id": mint_run_id(),
        "attempt_id": mint_attempt_id(),
        "execution_id": mint_execution_id(),
    }
    payload.update(overrides)
    return ExecutionProvenanceRef(**payload)


def _command(**overrides: object) -> CreateWorkItemExecutionLinkCommand:
    payload = {
        "tenant_id": _TENANT_A,
        "workspace_id": _WORKSPACE_A,
        "execution_link_id": "execution-link-1",
        "work_item_id": "work-item-1",
        "execution": _execution(),
        "linked_at": _CREATED_AT,
    }
    payload.update(overrides)
    return CreateWorkItemExecutionLinkCommand(**payload)


@pytest.fixture(params=("memory", "sqlite"))
def execution_link_repo(request: pytest.FixtureRequest, tmp_path: Path):
    if request.param == "memory":
        yield InMemoryWorkItemExecutionLinkRepository()
        return
    bundle = open_sqlite_collaborative_work_repositories(str(tmp_path / "execution-link.sqlite"))
    try:
        yield bundle.execution_link
    finally:
        bundle.close()


def test_repository_protocol_is_satisfied() -> None:
    repo = InMemoryWorkItemExecutionLinkRepository()
    assert isinstance(repo, WorkItemExecutionLinkRepository)


def test_sqlite_bundle_exposes_execution_link_port(tmp_path: Path) -> None:
    bundle = open_sqlite_collaborative_work_repositories(str(tmp_path / "bundle.sqlite"))
    try:
        assert isinstance(bundle.execution_link, WorkItemExecutionLinkRepository)
    finally:
        bundle.close()


def test_create_and_get(execution_link_repo: WorkItemExecutionLinkRepository) -> None:
    created = execution_link_repo.create(_command())
    loaded = execution_link_repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        execution_link_id="execution-link-1",
    )
    assert loaded == created
    assert loaded is not None
    assert loaded.execution.execution_id == created.execution.execution_id


def test_wrong_scope_returns_none(execution_link_repo: WorkItemExecutionLinkRepository) -> None:
    execution_link_repo.create(_command())
    assert (
        execution_link_repo.get(
            tenant_id=_TENANT_B,
            workspace_id=_WORKSPACE_B,
            execution_link_id="execution-link-1",
        )
        is None
    )


def test_duplicate_identity_raises(execution_link_repo: WorkItemExecutionLinkRepository) -> None:
    execution_link_repo.create(_command())
    with pytest.raises(WorkItemExecutionLinkAlreadyExists):
        execution_link_repo.create(_command())


def test_idempotency_replay_and_conflict(execution_link_repo: WorkItemExecutionLinkRepository) -> None:
    command = _command(idempotency_key="idem-key")
    created = execution_link_repo.create(command)
    replay = execution_link_repo.create(
        _command(
            idempotency_key="idem-key",
            linked_at=_LATER,
            execution=command.execution,
        ),
    )
    assert replay == created
    assert replay.linked_at == created.linked_at
    with pytest.raises(WorkItemExecutionLinkIdempotencyConflict):
        execution_link_repo.create(
            _command(
                execution_link_id="execution-link-2",
                idempotency_key="idem-key",
            ),
        )


def test_multiple_links_per_work_item(execution_link_repo: WorkItemExecutionLinkRepository) -> None:
    first = execution_link_repo.create(
        _command(execution_link_id="execution-link-a", execution=_execution()),
    )
    second = execution_link_repo.create(
        _command(execution_link_id="execution-link-b", execution=_execution()),
    )
    listed = execution_link_repo.list_for_work_item(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        work_item_id="work-item-1",
    )
    assert {first.execution_link_id, second.execution_link_id} == {
        record.execution_link_id for record in listed
    }


def test_list_ordering_is_linked_at_then_execution_link_id(
    execution_link_repo: WorkItemExecutionLinkRepository,
) -> None:
    first = execution_link_repo.create(
        _command(
            execution_link_id="execution-link-z",
            linked_at=_LATER,
            execution=_execution(),
        ),
    )
    second = execution_link_repo.create(
        _command(
            execution_link_id="execution-link-a",
            linked_at=_CREATED_AT,
            execution=_execution(),
        ),
    )
    third = execution_link_repo.create(
        _command(
            execution_link_id="execution-link-m",
            linked_at=_CREATED_AT,
            execution=_execution(),
        ),
    )
    listed = execution_link_repo.list_for_work_item(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        work_item_id="work-item-1",
    )
    assert listed == (second, third, first)


def test_malformed_execution_provenance_rejected() -> None:
    with pytest.raises(ValueError):
        ExecutionProvenanceRef(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id="not-canonical",
        )


def test_sqlite_durability_survives_restart(tmp_path: Path) -> None:
    db_path = str(tmp_path / "durability.sqlite")
    bundle = open_sqlite_collaborative_work_repositories(db_path)
    command = _command(idempotency_key="restart-idem")
    created = bundle.execution_link.create(command)
    bundle.close()

    reopened = open_sqlite_collaborative_work_repositories(db_path)
    try:
        loaded = reopened.execution_link.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            execution_link_id="execution-link-1",
        )
        assert loaded == created
        replay = reopened.execution_link.create(command)
        assert replay == created
        listed = reopened.execution_link.list_for_work_item(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_item_id="work-item-1",
        )
        assert listed == (created,)
    finally:
        reopened.close()
