# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import concurrent.futures

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_lineage import (
    ExecutionLineageAttemptClosureKind,
    ExecutionLineageIntegrityError,
    ExecutionLineagePersistence,
    build_execution_lineage_attempt_scope,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.execution.lineage.document_store_persistence import (
    DocumentStoreExecutionLineagePersistence,
)
from intergrax.runtime.execution.lineage.persistence import InMemoryExecutionLineagePersistence


def _scope(
    *,
    tenant_id: str = "tenant-a",
    task_id: str | None = None,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> object:
    return build_execution_lineage_attempt_scope(
        tenant_id=tenant_id,
        task_id=task_id or mint_task_id(),
        run_id=run_id or mint_run_id(),
        attempt_id=attempt_id or mint_attempt_id(),
    )


@pytest.fixture(params=["memory", "document_store"])
def persistence(request: pytest.FixtureRequest) -> ExecutionLineagePersistence:
    if request.param == "memory":
        return InMemoryExecutionLineagePersistence()
    return DocumentStoreExecutionLineagePersistence(InMemoryDocumentStore())


def test_open_attempt_idempotent(persistence: ExecutionLineagePersistence) -> None:
    scope = _scope()
    first = persistence.open_attempt(scope)
    second = persistence.open_attempt(scope)
    assert first == second


def test_root_and_child_admissions(persistence: ExecutionLineagePersistence) -> None:
    scope = _scope()
    root = mint_execution_id()
    persistence.open_attempt(scope)
    persistence.open_segment(scope, root)
    persistence.admit_root(scope, root, root)
    child = mint_execution_id()
    persistence.admit_child(scope, root, child, root)
    page = persistence.list_admissions_for_attempt(scope, limit=10)
    assert len(page.admissions) == 2
    assert page.admissions[0].admission_position == 1
    assert page.admissions[1].parent_execution_id == root


def test_duplicate_identical_child_is_idempotent(persistence: ExecutionLineagePersistence) -> None:
    scope = _scope()
    root = mint_execution_id()
    child = mint_execution_id()
    persistence.open_attempt(scope)
    persistence.open_segment(scope, root)
    persistence.admit_root(scope, root, root)
    first = persistence.admit_child(scope, root, child, root)
    second = persistence.admit_child(scope, root, child, root)
    assert first == second


def test_conflicting_parent_fails_closed(persistence: ExecutionLineagePersistence) -> None:
    scope = _scope()
    root = mint_execution_id()
    child = mint_execution_id()
    other_parent = mint_execution_id()
    persistence.open_attempt(scope)
    persistence.open_segment(scope, root)
    persistence.admit_root(scope, root, root)
    persistence.admit_child(scope, root, child, root)
    with pytest.raises(ExecutionLineageIntegrityError):
        persistence.admit_child(scope, root, child, other_parent)


def test_resume_segment_with_predecessor(persistence: ExecutionLineagePersistence) -> None:
    scope = _scope()
    e1 = mint_execution_id()
    e4 = mint_execution_id()
    persistence.open_attempt(scope)
    persistence.open_segment(scope, e1)
    persistence.admit_root(scope, e1, e1)
    persistence.close_segment_for_resume(scope, e1)
    segment = persistence.open_segment(scope, e4, e1)
    assert segment.predecessor_root_execution_id == e1


def test_unclean_resume_marks_degraded(persistence: ExecutionLineagePersistence) -> None:
    scope = _scope()
    e1 = mint_execution_id()
    e4 = mint_execution_id()
    persistence.open_attempt(scope)
    persistence.open_segment(scope, e1)
    persistence.open_segment(scope, e4, e1)
    state = persistence.read_attempt_lineage_state(scope)
    assert state is not None
    assert state.degraded is True
    assert state.active_segment_root_execution_id == e4


def test_seal_blocks_writes(persistence: ExecutionLineagePersistence) -> None:
    scope = _scope()
    root = mint_execution_id()
    persistence.open_attempt(scope)
    persistence.open_segment(scope, root)
    persistence.admit_root(scope, root, root)
    persistence.seal_attempt(scope, ExecutionLineageAttemptClosureKind.COMPLETED)
    with pytest.raises(ExecutionLineageIntegrityError):
        persistence.admit_child(scope, root, mint_execution_id(), root)


def test_concurrent_sibling_admissions(persistence: ExecutionLineagePersistence) -> None:
    scope = _scope()
    root = mint_execution_id()
    persistence.open_attempt(scope)
    persistence.open_segment(scope, root)
    persistence.admit_root(scope, root, root)

    def admit_one() -> None:
        persistence.admit_child(scope, root, mint_execution_id(), root)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(admit_one) for _ in range(32)]
        for future in futures:
            future.result()
    page = persistence.list_admissions_for_attempt(scope, limit=100)
    positions = [item.admission_position for item in page.admissions if item.execution_id != root]
    assert len(positions) == 32
    assert len(set(positions)) == 32


def test_pagination_stable_ordering(persistence: ExecutionLineagePersistence) -> None:
    scope = _scope()
    root = mint_execution_id()
    persistence.open_attempt(scope)
    persistence.open_segment(scope, root)
    persistence.admit_root(scope, root, root)
    for _ in range(5):
        persistence.admit_child(scope, root, mint_execution_id(), root)
    first = persistence.list_admissions_for_attempt(scope, limit=3)
    assert len(first.admissions) == 3
    second = persistence.list_admissions_for_attempt(
        scope,
        limit=3,
        cursor=first.next_cursor,
    )
    all_positions = [item.admission_position for item in first.admissions + second.admissions]
    assert all_positions == sorted(all_positions)
