# © Artur Czarnecki. All rights reserved.

"""DIAG-ENTERPRISE-1-R3 maintenance safety-age contract and recoverable projection health."""

from __future__ import annotations

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
    MIN_SAFE_PROJECTION_AGE,
    ProblemListIndexClassification,
    ProblemListIndexReconciliationError,
    ProblemListProjectionHealth,
    classify_list_index_projection,
    projection_age_is_below_destructive_threshold,
    resolve_effective_minimum_projection_age,
)
from intergrax.runtime.diagnostics.problem_list_query import (
    DecodedListIndexData,
    encode_list_index_data,
    list_index_row_key,
    list_scopes_for_status,
)
from intergrax.runtime.diagnostics.problem_persistence import ProblemPersistenceIntegrityError
from intergrax.runtime.diagnostics.problem_record_codec import encode_problem_record
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    document_store_problem_persistence_for_tests,
    in_memory_document_store_for_problem_tests,
)

pytestmark = pytest.mark.unit

_TENANT_A = "enterprise-1-r3-tenant-a"
_TENANT_B = "enterprise-1-r3-tenant-b"
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


def _seed_orphan(
    store,
    *,
    tenant: str,
    partition_key: str,
    age: timedelta,
) -> None:
    problem = replace(
        sample_problem(tenant_id=tenant, problem_id=mint_problem_id()),
        last_seen_at=_BASE_TIME + timedelta(days=1),
    )
    for scope in list_scopes_for_status(problem.status):
        store.put(
            _list_index_document(
                problem,
                scope=scope,
                partition_key=partition_key,
                projection_written_at=_NOW - age,
            ),
        )


def _complete_reconciliation(persistence, *, tenant: str) -> None:
    cursor: str | None = None
    while True:
        page = persistence.reconcile_list_indexes(
            tenant_id=tenant,
            minimum_projection_age=_MAINTENANCE_AGE,
            limit=500,
            cursor=cursor,
        )
        if not page.has_more:
            break
        cursor = page.next_cursor


def test_r2_bug_aggressive_cutoff_would_classify_orphan_but_safety_blocks() -> None:
    written_at = _NOW - timedelta(seconds=1)
    index = DecodedListIndexData(
        problem_id=mint_problem_id(),
        last_seen_at=_BASE_TIME,
        status=ProblemStatus.OPEN,
        record_version=1,
        projection_written_at=written_at,
        schema_version="intergrax.diagnostic_problem.list_index.v2",
    )
    # Old stale_before=T+1s contract would classify this as PROVEN_ORPHAN.
    assert (
        classify_list_index_projection(
            index=index,
            canonical=None,
            now=_NOW,
            minimum_projection_age=MIN_SAFE_PROJECTION_AGE,
        )
        is ProblemListIndexClassification.TRANSIENT_OR_UNCERTAIN
    )


def test_zero_age_attack_rejected() -> None:
    with pytest.raises(ProblemListIndexReconciliationError, match="minimum_projection_age"):
        resolve_effective_minimum_projection_age(minimum_projection_age=timedelta(0))

    persistence = _persistence_with_clock()
    with pytest.raises(ProblemListIndexReconciliationError, match="minimum_projection_age"):
        persistence.reconcile_list_indexes(
            tenant_id=_TENANT_A,
            minimum_projection_age=timedelta(0),
            limit=10,
        )


def test_below_platform_minimum_rejected() -> None:
    with pytest.raises(ProblemListIndexReconciliationError):
        resolve_effective_minimum_projection_age(
            minimum_projection_age=MIN_SAFE_PROJECTION_AGE - timedelta(seconds=1),
        )


def test_future_projection_is_transient() -> None:
    future_written_at = _NOW + timedelta(minutes=1)
    index = DecodedListIndexData(
        problem_id=mint_problem_id(),
        last_seen_at=_BASE_TIME,
        status=ProblemStatus.OPEN,
        record_version=1,
        projection_written_at=future_written_at,
        schema_version="intergrax.diagnostic_problem.list_index.v2",
    )
    assert (
        classify_list_index_projection(
            index=index,
            canonical=None,
            now=_NOW,
            minimum_projection_age=MIN_SAFE_PROJECTION_AGE,
        )
        is ProblemListIndexClassification.TRANSIENT_OR_UNCERTAIN
    )
    assert projection_age_is_below_destructive_threshold(
        projection_written_at=future_written_at,
        now=_NOW,
        minimum_projection_age=MIN_SAFE_PROJECTION_AGE,
    )


def test_exact_safety_boundary_one_microsecond_below_is_transient() -> None:
    written_at = _NOW - MIN_SAFE_PROJECTION_AGE + timedelta(microseconds=1)
    assert projection_age_is_below_destructive_threshold(
        projection_written_at=written_at,
        now=_NOW,
        minimum_projection_age=MIN_SAFE_PROJECTION_AGE,
    )


def test_exact_safety_boundary_at_minimum_is_transient() -> None:
    written_at = _NOW - MIN_SAFE_PROJECTION_AGE
    assert projection_age_is_below_destructive_threshold(
        projection_written_at=written_at,
        now=_NOW,
        minimum_projection_age=MIN_SAFE_PROJECTION_AGE,
    )


def test_beyond_minimum_is_eligible_for_orphan() -> None:
    written_at = _NOW - MIN_SAFE_PROJECTION_AGE - timedelta(microseconds=1)
    index = DecodedListIndexData(
        problem_id=mint_problem_id(),
        last_seen_at=_BASE_TIME,
        status=ProblemStatus.OPEN,
        record_version=1,
        projection_written_at=written_at,
        schema_version="intergrax.diagnostic_problem.list_index.v2",
    )
    assert (
        classify_list_index_projection(
            index=index,
            canonical=None,
            now=_NOW,
            minimum_projection_age=MIN_SAFE_PROJECTION_AGE,
        )
        is ProblemListIndexClassification.PROVEN_ORPHAN
    )


def test_active_writer_survives_aggressive_maintenance() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    problem = sample_problem(tenant_id=_TENANT_A, problem_id=mint_problem_id())
    for scope in list_scopes_for_status(problem.status):
        store.put(
            _list_index_document(
                problem,
                scope=scope,
                partition_key=_PARTITION_A,
                projection_written_at=_NOW - timedelta(seconds=30),
            ),
        )

    page = persistence.reconcile_list_indexes(
        tenant_id=_TENANT_A,
        minimum_projection_age=MIN_SAFE_PROJECTION_AGE,
        limit=100,
    )
    assert page.deleted == 0
    assert page.transient >= 1

    store.put(
        DocumentRecord(
            partition_key=_PARTITION_A,
            row_key=f"record:{problem.problem_id}",
            data=encode_problem_record(problem),
        ),
    )
    listed = persistence.query_problems(tenant_id=_TENANT_A, limit=10)
    assert len(listed.problems) == 1
    assert listed.problems[0].problem_id == problem.problem_id


def test_old_orphan_deleted_after_safety_age() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    _seed_orphan(
        store,
        tenant=_TENANT_A,
        partition_key=_PARTITION_A,
        age=_MAINTENANCE_AGE + timedelta(minutes=5),
    )

    page = persistence.reconcile_list_indexes(
        tenant_id=_TENANT_A,
        minimum_projection_age=_MAINTENANCE_AGE,
        limit=100,
    )
    assert page.deleted >= 1


def test_old_stale_repaired_after_safety_age() -> None:
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

    page = persistence.reconcile_list_indexes(
        tenant_id=_TENANT_A,
        minimum_projection_age=_MAINTENANCE_AGE,
        limit=100,
    )
    assert page.repaired >= len(list_scopes_for_status(ahead.status))


def test_health_h1_clean_is_healthy() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    problem = sample_problem(tenant_id=_TENANT_A, problem_id=mint_problem_id())
    persistence.create(problem)

    _complete_reconciliation(persistence, tenant=_TENANT_A)
    assert persistence.projection_health() is ProblemListProjectionHealth.HEALTHY


def test_health_h2_many_skips_then_clean_query_recovers() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    for index in range(12):
        problem = sample_problem(
            tenant_id=_TENANT_A,
            problem_id=mint_problem_id(),
        )
        for scope in list_scopes_for_status(problem.status):
            store.put(
                _list_index_document(
                    problem,
                    scope=scope,
                    partition_key=_PARTITION_A,
                    projection_written_at=_NOW - _MAINTENANCE_AGE,
                ),
            )

    persistence.query_problems(tenant_id=_TENANT_A, limit=1)
    assert persistence.projection_health() is ProblemListProjectionHealth.DEGRADED

    clean = sample_problem(tenant_id=_TENANT_A, problem_id=mint_problem_id())
    persistence.create(clean)
    persistence.query_problems(tenant_id=_TENANT_A, limit=1)
    assert persistence.projection_health() is ProblemListProjectionHealth.HEALTHY


def test_health_h3_maintenance_repair_degrades() -> None:
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


def test_health_h4_complete_clean_reconciliation_recovers() -> None:
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


def test_health_h5_corruption_stays_degraded() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    problem = sample_problem(tenant_id=_TENANT_A, problem_id=mint_problem_id())
    persistence.create(problem)
    corrupted = replace(problem, last_seen_at=_BASE_TIME + timedelta(days=3))
    for scope in list_scopes_for_status(corrupted.status):
        store.put(
            _list_index_document(
                corrupted,
                scope=scope,
                partition_key=_PARTITION_A,
                projection_written_at=_NOW - _MAINTENANCE_AGE,
            ),
        )

    with pytest.raises(ProblemPersistenceIntegrityError):
        persistence.query_problems(tenant_id=_TENANT_A, limit=5)
    assert persistence.projection_health() is ProblemListProjectionHealth.DEGRADED

    _complete_reconciliation(persistence, tenant=_TENANT_A)
    assert persistence.projection_health() is ProblemListProjectionHealth.DEGRADED


def test_health_h6_unrelated_tenant_clean_scan_does_not_hide_corruption() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)

    corrupted = sample_problem(tenant_id=_TENANT_A, problem_id=mint_problem_id())
    persistence.create(corrupted)
    bad = replace(corrupted, last_seen_at=_BASE_TIME + timedelta(days=3))
    for scope in list_scopes_for_status(bad.status):
        store.put(
            _list_index_document(
                bad,
                scope=scope,
                partition_key=_PARTITION_A,
                projection_written_at=_NOW - _MAINTENANCE_AGE,
            ),
        )
    with pytest.raises(ProblemPersistenceIntegrityError):
        persistence.query_problems(tenant_id=_TENANT_A, limit=5)

    clean = sample_problem(tenant_id=_TENANT_B, problem_id=mint_problem_id())
    persistence.create(clean)
    _complete_reconciliation(persistence, tenant=_TENANT_B)

    assert persistence.projection_health() is ProblemListProjectionHealth.DEGRADED
