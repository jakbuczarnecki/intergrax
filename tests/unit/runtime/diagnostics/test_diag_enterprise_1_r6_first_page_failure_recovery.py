# © Artur Czarnecki. All rights reserved.

"""DIAG-ENTERPRISE-1-R6 first-page maintenance cycle failure recovery."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from intergrax.applications._shared.diagnostic_cursor_secret import (  # noqa: F401
    resolve_problem_list_cursor_secret,
)
from intergrax.integrations._shared.in_memory_document_store import (  # noqa: F401
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.runtime.diagnostics.persistence_conformance import sample_problem
from intergrax.runtime.diagnostics.problem_lifecycle import mint_problem_id
from intergrax.runtime.diagnostics.problem_list_index_reconciliation import (
    ProblemListMaintenanceCycleKey,
    ProblemListProjectionHealth,
)
from intergrax.runtime.diagnostics.problem_list_query import (
    encode_list_index_data,
    list_index_row_key,
    list_scopes_for_status,
)
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    TEST_DOCUMENT_STORE_CURSOR_SECRET,
    document_store_problem_persistence_for_tests,
    in_memory_document_store_for_problem_tests,
)

pytestmark = pytest.mark.unit

_TENANT_A = "enterprise-1-r6-tenant-a"
_BASE_TIME = datetime(2026, 8, 26, 9, 0, tzinfo=UTC)
_NOW = datetime(2026, 8, 31, 12, 0, tzinfo=UTC)
_MAINTENANCE_AGE = timedelta(hours=1)
_PARTITION_A = f"intergrax.diagnostic_problem.v1:{_TENANT_A}"


def _persistence_with_clock(store=None):
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


def _cycle_state(persistence, *, tenant: str = _TENANT_A):
    reconciler = persistence._list_index_reconciler
    cycle_key = ProblemListMaintenanceCycleKey(tenant_id=tenant, scope=None)
    with reconciler.health_state._lock:
        return replace(reconciler.health_state.maintenance_cycles[cycle_key])


def _registry_size(persistence) -> int:
    health_state = persistence._list_index_reconciler.health_state
    with health_state._lock:
        return len(health_state.maintenance_cycles)


class _FailingFirstQueryStore(InMemoryDocumentStore):
    """Raises on the first cursor=None query only."""

    def __init__(self) -> None:
        super().__init__(cursor_secret=TEST_DOCUMENT_STORE_CURSOR_SECRET)
        self._fail_next_first_query = False

    def query(self, partition_key, *, limit, row_key_prefix=None, cursor=None):
        if cursor is None and self._fail_next_first_query:
            self._fail_next_first_query = False
            raise RuntimeError("document store query failed")
        return super().query(
            partition_key,
            limit=limit,
            row_key_prefix=row_key_prefix,
            cursor=cursor,
        )

    def arm_next_first_query_failure(self) -> None:
        self._fail_next_first_query = True


class _FailingContinuationQueryStore(InMemoryDocumentStore):
    """Raises on the first continuation query only."""

    def __init__(self) -> None:
        super().__init__(cursor_secret=TEST_DOCUMENT_STORE_CURSOR_SECRET)
        self._fail_next_continuation = False

    def query(self, partition_key, *, limit, row_key_prefix=None, cursor=None):
        if cursor is not None and self._fail_next_continuation:
            self._fail_next_continuation = False
            raise RuntimeError("document store query failed")
        return super().query(
            partition_key,
            limit=limit,
            row_key_prefix=row_key_prefix,
            cursor=cursor,
        )

    def arm_next_continuation_failure(self) -> None:
        self._fail_next_continuation = True


def test_fp1_fresh_cycle_first_query_failure_retry_succeeds() -> None:
    store = _FailingFirstQueryStore()
    persistence = _persistence_with_clock(store)
    _seed_orphans(store, tenant=_TENANT_A, partition_key=_PARTITION_A, count=2)
    store.arm_next_first_query_failure()

    with pytest.raises(RuntimeError, match="document store query failed"):
        persistence.reconcile_list_indexes(
            tenant_id=_TENANT_A,
            minimum_projection_age=_MAINTENANCE_AGE,
        )

    assert _registry_size(persistence) == 0

    page = persistence.reconcile_list_indexes(
        tenant_id=_TENANT_A,
        minimum_projection_age=_MAINTENANCE_AGE,
        limit=100,
    )
    assert page.deleted >= 1


def test_fp2_fresh_cycle_failure_after_partial_repair_retry_converges() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    _seed_orphans(store, tenant=_TENANT_A, partition_key=_PARTITION_A, count=3)
    reconciler = persistence._list_index_reconciler
    calls = {"count": 0}
    original_reconcile_one = reconciler._reconcile_one

    def failing_second(*, index_document, tenant_id, partition_key, now, minimum_projection_age):
        calls["count"] += 1
        if calls["count"] >= 2:
            raise RuntimeError("reconcile operation failed")
        return original_reconcile_one(
            index_document=index_document,
            tenant_id=tenant_id,
            partition_key=partition_key,
            now=now,
            minimum_projection_age=minimum_projection_age,
        )

    with patch.object(reconciler, "_reconcile_one", side_effect=failing_second):
        with pytest.raises(RuntimeError, match="reconcile operation failed"):
            persistence.reconcile_list_indexes(
                tenant_id=_TENANT_A,
                minimum_projection_age=_MAINTENANCE_AGE,
                limit=10,
            )

    telemetry = persistence.projection_telemetry_snapshot()
    assert telemetry.deleted_orphan_projection >= 1
    assert persistence.projection_health() is ProblemListProjectionHealth.DEGRADED

    page = persistence.reconcile_list_indexes(
        tenant_id=_TENANT_A,
        minimum_projection_age=_MAINTENANCE_AGE,
        limit=100,
    )
    assert page.has_more is False
    assert page.deleted + page.repaired + page.consistent >= 1


def test_fp3_degraded_retained_state_recovery_first_page_fails() -> None:
    store = _FailingFirstQueryStore()
    persistence = _persistence_with_clock(store)
    _seed_orphans(store, tenant=_TENANT_A, partition_key=_PARTITION_A, count=2)

    completed = persistence.reconcile_list_indexes(
        tenant_id=_TENANT_A,
        minimum_projection_age=_MAINTENANCE_AGE,
        limit=100,
    )
    assert completed.deleted >= 1
    assert completed.has_more is False
    assert persistence.projection_health() is ProblemListProjectionHealth.DEGRADED

    store.arm_next_first_query_failure()
    with pytest.raises(RuntimeError, match="document store query failed"):
        persistence.reconcile_list_indexes(
            tenant_id=_TENANT_A,
            minimum_projection_age=_MAINTENANCE_AGE,
            limit=1,
        )

    assert persistence.projection_health() is ProblemListProjectionHealth.DEGRADED


def test_fp4_successful_first_partial_page_keeps_cycle_in_progress() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    _seed_orphans(store, tenant=_TENANT_A, partition_key=_PARTITION_A, count=3)

    page = persistence.reconcile_list_indexes(
        tenant_id=_TENANT_A,
        minimum_projection_age=_MAINTENANCE_AGE,
        limit=1,
    )
    assert page.has_more is True
    assert page.next_cursor is not None

    state = _cycle_state(persistence)
    assert state.in_progress is True
    assert state.page_in_flight is False

    continuation = persistence.reconcile_list_indexes(
        tenant_id=_TENANT_A,
        minimum_projection_age=_MAINTENANCE_AGE,
        limit=100,
        cursor=page.next_cursor,
    )
    assert continuation.examined >= 1


def test_fp5_successful_first_complete_page_prunes_registry() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    persistence.create(sample_problem(tenant_id=_TENANT_A, problem_id=mint_problem_id()))

    page = persistence.reconcile_list_indexes(
        tenant_id=_TENANT_A,
        minimum_projection_age=_MAINTENANCE_AGE,
        limit=100,
    )
    assert page.has_more is False
    assert persistence.projection_health() is ProblemListProjectionHealth.HEALTHY
    assert _registry_size(persistence) == 0


def test_fp6_continuation_exception_retry_same_cursor_succeeds() -> None:
    store = _FailingContinuationQueryStore()
    persistence = _persistence_with_clock(store)
    _seed_orphans(store, tenant=_TENANT_A, partition_key=_PARTITION_A, count=2)
    first = persistence.reconcile_list_indexes(
        tenant_id=_TENANT_A,
        minimum_projection_age=_MAINTENANCE_AGE,
        limit=1,
    )
    cursor = first.next_cursor
    assert cursor is not None
    store.arm_next_continuation_failure()

    with pytest.raises(RuntimeError, match="document store query failed"):
        persistence.reconcile_list_indexes(
            tenant_id=_TENANT_A,
            minimum_projection_age=_MAINTENANCE_AGE,
            limit=1,
            cursor=cursor,
        )

    state = _cycle_state(persistence)
    assert state.in_progress is True
    assert state.page_in_flight is False

    page = persistence.reconcile_list_indexes(
        tenant_id=_TENANT_A,
        minimum_projection_age=_MAINTENANCE_AGE,
        limit=1,
        cursor=cursor,
    )
    assert page.examined >= 1


def test_fp7_first_page_exception_releases_page_in_flight() -> None:
    store = _FailingFirstQueryStore()
    persistence = _persistence_with_clock(store)
    _seed_orphans(store, tenant=_TENANT_A, partition_key=_PARTITION_A, count=2)
    store.arm_next_first_query_failure()

    with pytest.raises(RuntimeError, match="document store query failed"):
        persistence.reconcile_list_indexes(
            tenant_id=_TENANT_A,
            minimum_projection_age=_MAINTENANCE_AGE,
        )

    with pytest.raises(RuntimeError, match="document store query failed"):
        store.arm_next_first_query_failure()
        persistence.reconcile_list_indexes(
            tenant_id=_TENANT_A,
            minimum_projection_age=_MAINTENANCE_AGE,
        )

    page = persistence.reconcile_list_indexes(
        tenant_id=_TENANT_A,
        minimum_projection_age=_MAINTENANCE_AGE,
        limit=100,
    )
    assert page.deleted >= 1
