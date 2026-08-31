# © Artur Czarnecki. All rights reserved.

"""DIAG-ENTERPRISE-1-R4 maintenance cycle identity and projection health correctness."""

from __future__ import annotations

import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.applications._shared.diagnostic_cursor_secret import (  # noqa: F401
    resolve_problem_list_cursor_secret,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore  # noqa: F401
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.runtime.diagnostics.persistence_conformance import sample_problem
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemStatus, mint_problem_id
from intergrax.runtime.diagnostics.problem_list_index_reconciliation import (
    ProblemListIndexReconciliationError,
    ProblemListProjectionHealth,
)
from intergrax.runtime.diagnostics.problem_list_query import (
    ProblemListScope,
    encode_list_index_data,
    list_index_row_key,
    list_scopes_for_status,
)
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    document_store_problem_persistence_for_tests,
    in_memory_document_store_for_problem_tests,
)

pytestmark = pytest.mark.unit

_TENANT_A = "enterprise-1-r4-tenant-a"
_TENANT_B = "enterprise-1-r4-tenant-b"
_BASE_TIME = datetime(2026, 8, 26, 9, 0, tzinfo=UTC)
_NOW = datetime(2026, 8, 31, 12, 0, tzinfo=UTC)
_MAINTENANCE_AGE = timedelta(hours=1)
_PARTITION_A = f"intergrax.diagnostic_problem.v1:{_TENANT_A}"
_PARTITION_B = f"intergrax.diagnostic_problem.v1:{_TENANT_B}"


def _persistence_with_clock(store=None, *, tenant: str = _TENANT_A):
    persistence = document_store_problem_persistence_for_tests(store)
    persistence.set_clock_for_tests(lambda: _NOW)
    return persistence


def _list_index_document(
    problem,
    *,
    scope,
    partition_key: str,
    projection_written_at: datetime | None = None,
):
    return DocumentRecord(
        partition_key=partition_key,
        row_key=list_index_row_key(scope=scope, problem=problem),
        data=encode_list_index_data(
            problem_id=problem.problem_id,
            last_seen_at=problem.last_seen_at,
            status=problem.status,
            record_version=problem.record_version,
            projection_written_at=projection_written_at,
        ),
    )


def _seed_orphans(
    store,
    *,
    tenant: str,
    partition_key: str,
    count: int,
) -> None:
    for index in range(count):
        problem = replace(
            sample_problem(tenant_id=tenant, problem_id=mint_problem_id()),
            last_seen_at=_BASE_TIME + timedelta(seconds=index),
        )
        for scope in list_scopes_for_status(problem.status):
            store.put(
                _list_index_document(
                    problem,
                    scope=scope,
                    partition_key=partition_key,
                    projection_written_at=_NOW - _MAINTENANCE_AGE - timedelta(minutes=5),
                ),
            )


def _complete_reconciliation(
    persistence,
    *,
    tenant: str,
    scope: ProblemListScope | None = None,
) -> None:
    cursor: str | None = None
    while True:
        page = persistence.reconcile_list_indexes(
            tenant_id=tenant,
            minimum_projection_age=_MAINTENANCE_AGE,
            scope=scope,
            limit=500,
            cursor=cursor,
        )
        if not page.has_more:
            break
        cursor = page.next_cursor


def _registry_size(persistence) -> int:
    health_state = persistence._list_index_reconciler.health_state
    with health_state._lock:
        return len(health_state.maintenance_cycles)


def test_c1_abandoned_tenant_a_clean_tenant_b_stays_degraded() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    _seed_orphans(
        store,
        tenant=_TENANT_A,
        partition_key=_PARTITION_A,
        count=3,
    )

    page_a = persistence.reconcile_list_indexes(
        tenant_id=_TENANT_A,
        minimum_projection_age=_MAINTENANCE_AGE,
        limit=1,
    )
    assert page_a.has_more is True
    assert page_a.deleted + page_a.repaired + page_a.corrupt >= 1
    assert persistence.projection_health() is ProblemListProjectionHealth.DEGRADED

    clean_b = sample_problem(tenant_id=_TENANT_B, problem_id=mint_problem_id())
    persistence.create(clean_b)
    _complete_reconciliation(persistence, tenant=_TENANT_B)

    assert persistence.projection_health() is ProblemListProjectionHealth.DEGRADED


def test_c2_degraded_open_scope_not_hidden_by_clean_resolved() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    open_problem = replace(
        sample_problem(tenant_id=_TENANT_A, problem_id=mint_problem_id()),
        status=ProblemStatus.OPEN,
    )
    store.put(
        _list_index_document(
            open_problem,
            scope=ProblemListScope.OPEN,
            partition_key=_PARTITION_A,
            projection_written_at=_NOW - _MAINTENANCE_AGE - timedelta(minutes=5),
        ),
    )

    page_open = persistence.reconcile_list_indexes(
        tenant_id=_TENANT_A,
        minimum_projection_age=_MAINTENANCE_AGE,
        scope=ProblemListScope.OPEN,
        limit=100,
    )
    assert page_open.deleted >= 1
    assert persistence.projection_health() is ProblemListProjectionHealth.DEGRADED

    resolved = replace(
        sample_problem(tenant_id=_TENANT_A, problem_id=mint_problem_id()),
        status=ProblemStatus.RESOLVED,
    )
    persistence.create(resolved)
    _complete_reconciliation(
        persistence,
        tenant=_TENANT_A,
        scope=ProblemListScope.RESOLVED,
    )

    assert persistence.projection_health() is ProblemListProjectionHealth.DEGRADED


def test_c3_same_key_full_clean_recovery_becomes_healthy() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    problem = sample_problem(tenant_id=_TENANT_A, problem_id=mint_problem_id())
    persistence.create(problem)
    ahead = replace(
        problem,
        last_seen_at=_BASE_TIME + timedelta(hours=1),
        record_version=problem.record_version + 1,
    )
    for scope in list_scopes_for_status(ahead.status):
        store.put(
            _list_index_document(
                ahead,
                scope=scope,
                partition_key=_PARTITION_A,
                projection_written_at=_NOW - _MAINTENANCE_AGE - timedelta(minutes=5),
            ),
        )

    persistence.reconcile_list_indexes(
        tenant_id=_TENANT_A,
        minimum_projection_age=_MAINTENANCE_AGE,
        limit=100,
    )
    assert persistence.projection_health() is ProblemListProjectionHealth.DEGRADED

    _complete_reconciliation(persistence, tenant=_TENANT_A)
    assert persistence.projection_health() is ProblemListProjectionHealth.HEALTHY


def test_c4_two_independent_clean_cycles_are_healthy() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    persistence.create(sample_problem(tenant_id=_TENANT_A, problem_id=mint_problem_id()))
    persistence.create(sample_problem(tenant_id=_TENANT_B, problem_id=mint_problem_id()))

    _complete_reconciliation(persistence, tenant=_TENANT_A)
    _complete_reconciliation(persistence, tenant=_TENANT_B)

    assert persistence.projection_health() is ProblemListProjectionHealth.HEALTHY


def test_c5_active_cycle_cannot_be_silent_reset_by_cursor_none() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    _seed_orphans(
        store,
        tenant=_TENANT_A,
        partition_key=_PARTITION_A,
        count=3,
    )

    first = persistence.reconcile_list_indexes(
        tenant_id=_TENANT_A,
        minimum_projection_age=_MAINTENANCE_AGE,
        limit=1,
    )
    assert first.has_more is True

    with pytest.raises(ProblemListIndexReconciliationError, match="continuation cursor required"):
        persistence.reconcile_list_indexes(
            tenant_id=_TENANT_A,
            minimum_projection_age=_MAINTENANCE_AGE,
            limit=1,
        )

    assert persistence.projection_health() is ProblemListProjectionHealth.DEGRADED


def test_c6_completed_clean_cycles_do_not_grow_registry_without_bound() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)

    for index in range(40):
        tenant = f"enterprise-1-r4-prune-{index}"
        persistence.create(
            sample_problem(tenant_id=tenant, problem_id=mint_problem_id()),
        )
        _complete_reconciliation(persistence, tenant=tenant)

    assert _registry_size(persistence) == 0


def test_cursor_misuse_across_scopes_still_fails() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    _seed_orphans(
        store,
        tenant=_TENANT_A,
        partition_key=_PARTITION_A,
        count=2,
    )

    open_page = persistence.reconcile_list_indexes(
        tenant_id=_TENANT_A,
        minimum_projection_age=_MAINTENANCE_AGE,
        scope=ProblemListScope.OPEN,
        limit=1,
    )
    assert open_page.next_cursor is not None

    with pytest.raises(ValueError, match="document_store_cursor_query_mismatch"):
        persistence.reconcile_list_indexes(
            tenant_id=_TENANT_A,
            minimum_projection_age=_MAINTENANCE_AGE,
            scope=ProblemListScope.RESOLVED,
            limit=1,
            cursor=open_page.next_cursor,
        )


def test_concurrent_cycles_preserve_issue_state() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    _seed_orphans(
        store,
        tenant=_TENANT_A,
        partition_key=_PARTITION_A,
        count=4,
    )
    _seed_orphans(
        store,
        tenant=_TENANT_B,
        partition_key=_PARTITION_B,
        count=4,
    )

    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def run_partial(tenant: str) -> None:
        try:
            barrier.wait(timeout=5)
            persistence.reconcile_list_indexes(
                tenant_id=tenant,
                minimum_projection_age=_MAINTENANCE_AGE,
                limit=1,
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = (
        threading.Thread(target=run_partial, args=(_TENANT_A,)),
        threading.Thread(target=run_partial, args=(_TENANT_B,)),
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert persistence.projection_health() is ProblemListProjectionHealth.DEGRADED
