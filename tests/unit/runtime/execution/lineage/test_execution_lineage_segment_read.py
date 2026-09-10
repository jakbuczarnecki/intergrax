# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id, mint_run_id, mint_task_id
from intergrax.contracts.execution_lineage import ExecutionLineagePersistence, build_execution_lineage_attempt_scope
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.execution.lineage.document_store_persistence import DocumentStoreExecutionLineagePersistence
from intergrax.runtime.execution.lineage.persistence import InMemoryExecutionLineagePersistence
from tests.unit.runtime.execution.lineage.lineage_test_helpers import register_v1_attempt


def _scope() -> object:
    return build_execution_lineage_attempt_scope(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )


@pytest.fixture(params=["memory", "document_store"])
def persistence(request: pytest.FixtureRequest) -> ExecutionLineagePersistence:
    if request.param == "memory":
        return InMemoryExecutionLineagePersistence()
    return DocumentStoreExecutionLineagePersistence(InMemoryDocumentStore())


def test_list_segments_for_attempt_bounded_pagination(
    persistence: ExecutionLineagePersistence,
) -> None:
    scope = _scope()
    roots = [mint_execution_id() for _ in range(3)]
    register_v1_attempt(persistence, scope)
    predecessor: str | None = None
    for root in roots:
        persistence.open_segment(scope, root, predecessor)
        persistence.admit_root(scope, root, root)
        predecessor = root

    first = persistence.list_segments_for_attempt(scope, limit=2)
    assert len(first.segments) == 2
    assert first.next_cursor is not None

    second = persistence.list_segments_for_attempt(
        scope,
        limit=2,
        cursor=first.next_cursor,
    )
    assert len(second.segments) == 1
    assert second.next_cursor is None

    seen_roots = {segment.root_execution_id for segment in first.segments}
    seen_roots.update(segment.root_execution_id for segment in second.segments)
    assert seen_roots == set(roots)


def test_list_segments_tenant_isolation(
    persistence: ExecutionLineagePersistence,
) -> None:
    scope_a = build_execution_lineage_attempt_scope(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )
    scope_b = build_execution_lineage_attempt_scope(
        tenant_id="tenant-b",
        task_id=scope_a.task_id,
        run_id=scope_a.run_id,
        attempt_id=scope_a.attempt_id,
    )
    root = mint_execution_id()
    register_v1_attempt(persistence, scope_a)
    persistence.open_segment(scope_a, root)
    persistence.admit_root(scope_a, root, root)

    assert persistence.list_segments_for_attempt(scope_b, limit=10).segments == ()
