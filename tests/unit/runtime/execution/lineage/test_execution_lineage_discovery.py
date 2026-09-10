# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import concurrent.futures

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_lineage import (
    ExecutionLineageConfigurationError,
    ExecutionLineageIntegrityError,
    ExecutionLineagePersistence,
    build_execution_lineage_attempt_scope,
    build_execution_lineage_run_scope,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.execution.lineage.document_store_persistence import (
    DocumentStoreExecutionLineagePersistence,
)
from intergrax.runtime.execution.lineage.persistence import InMemoryExecutionLineagePersistence
from intergrax.runtime.execution.lineage.root_activation import (
    activate_root_execution_lineage,
    deactivate_root_execution_lineage,
)


def _scope(*, tenant_id: str = "tenant-a", task_id=None, run_id=None, attempt_id=None):
    return build_execution_lineage_attempt_scope(
        tenant_id=tenant_id,
        task_id=task_id or mint_task_id(),
        run_id=run_id or mint_run_id(),
        attempt_id=attempt_id or mint_attempt_id(),
    )


def _run_scope(scope) -> object:
    return build_execution_lineage_run_scope(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    )


@pytest.fixture(params=["memory", "document_store"])
def persistence(request: pytest.FixtureRequest) -> ExecutionLineagePersistence:
    if request.param == "memory":
        return InMemoryExecutionLineagePersistence()
    return DocumentStoreExecutionLineagePersistence(InMemoryDocumentStore())


def test_d1_index_first_root_activation(persistence: ExecutionLineagePersistence) -> None:
    scope = _scope()
    run_scope = _run_scope(scope)
    from intergrax.contracts.execution_identity import mint_execution_id

    root = mint_execution_id()
    state, token, degradation = activate_root_execution_lineage(
        persistence=persistence,
        scope=scope,
        root_execution_id=root,
    )
    deactivate_root_execution_lineage(token, degradation)
    record = persistence.read_attempt_discovery_record(run_scope, scope.attempt_id)
    assert record is not None
    assert record.discovery_position == 1
    attempt_state = persistence.read_attempt_lineage_state(scope)
    assert attempt_state is not None
    assert attempt_state.discovery_contract_version == 1


def test_d2_registration_idempotent(persistence: ExecutionLineagePersistence) -> None:
    scope = _scope()
    run_scope = _run_scope(scope)
    first = persistence.register_attempt_for_run(run_scope, scope.attempt_id)
    second = persistence.register_attempt_for_run(run_scope, scope.attempt_id)
    assert first == second
    run_state = persistence.read_discovery_run_state(run_scope)
    assert run_state is not None
    assert run_state.next_discovery_position == 2


def test_d3_concurrent_same_attempt(persistence: ExecutionLineagePersistence) -> None:
    scope = _scope()
    run_scope = _run_scope(scope)

    def register_once() -> object:
        return persistence.register_attempt_for_run(run_scope, scope.attempt_id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = [future.result() for future in [pool.submit(register_once) for _ in range(8)]]
    assert len({item.discovery_position for item in results}) == 1


def test_d4_concurrent_different_attempts(persistence: ExecutionLineagePersistence) -> None:
    run_id = mint_run_id()
    task_id = mint_task_id()
    scopes = [
        _scope(task_id=task_id, run_id=run_id, attempt_id=mint_attempt_id())
        for _ in range(8)
    ]
    run_scope = build_execution_lineage_run_scope(
        tenant_id="tenant-a",
        task_id=task_id,
        run_id=run_id,
    )

    def register(scope) -> object:
        return persistence.register_attempt_for_run(run_scope, scope.attempt_id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        records = [future.result() for future in [pool.submit(register, s) for s in scopes]]
    positions = [record.discovery_position for record in records]
    assert len(set(positions)) == len(positions)
    assert positions == list(range(1, len(positions) + 1))


def test_d5_crash_after_register_before_open(persistence: ExecutionLineagePersistence) -> None:
    scope = _scope()
    run_scope = _run_scope(scope)
    persistence.register_attempt_for_run(run_scope, scope.attempt_id)
    record = persistence.read_attempt_discovery_record(run_scope, scope.attempt_id)
    assert record is not None
    assert persistence.read_attempt_lineage_state(scope) is None


def test_d6_post_v1_open_without_discovery_row(persistence: ExecutionLineagePersistence) -> None:
    scope = _scope()
    with pytest.raises(ExecutionLineageIntegrityError):
        persistence.open_attempt(scope, discovery_contract_version=1)


def test_d7_new_legacy_open_blocked(persistence: ExecutionLineagePersistence) -> None:
    scope = _scope()
    with pytest.raises(ExecutionLineageConfigurationError):
        persistence.open_attempt(scope, discovery_contract_version=None)


def test_d10_paginated_run_discovery(persistence: ExecutionLineagePersistence) -> None:
    run_id = mint_run_id()
    task_id = mint_task_id()
    run_scope = build_execution_lineage_run_scope(
        tenant_id="tenant-a",
        task_id=task_id,
        run_id=run_id,
    )
    attempt_ids = [mint_attempt_id() for _ in range(5)]
    for attempt_id in attempt_ids:
        persistence.register_attempt_for_run(run_scope, attempt_id)
    first = persistence.list_attempts_for_run(run_scope, limit=2)
    assert len(first.attempts) == 2
    second = persistence.list_attempts_for_run(
        run_scope,
        limit=2,
        cursor=first.next_cursor,
    )
    all_positions = [
        item.discovery_position for item in first.attempts + second.attempts
    ]
    assert all_positions == [1, 2, 3, 4]
