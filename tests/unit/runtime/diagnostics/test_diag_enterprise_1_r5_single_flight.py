# © Artur Czarnecki. All rights reserved.

"""DIAG-ENTERPRISE-1-R5 single-flight maintenance page ownership."""

from __future__ import annotations

import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta

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
    TEST_DOCUMENT_STORE_CURSOR_SECRET,
    document_store_problem_persistence_for_tests,
    in_memory_document_store_for_problem_tests,
)

pytestmark = pytest.mark.unit

_TENANT_A = "enterprise-1-r5-tenant-a"
_TENANT_B = "enterprise-1-r5-tenant-b"
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


def _registry_size(persistence) -> int:
    health_state = persistence._list_index_reconciler.health_state
    with health_state._lock:
        return len(health_state.maintenance_cycles)


class _FailingQueryStore(InMemoryDocumentStore):
    """Raises on the first continuation query only."""

    def __init__(self) -> None:
        super().__init__(cursor_secret=TEST_DOCUMENT_STORE_CURSOR_SECRET)
        self._fail_next_continuation = False

    def query(self, partition_key, *, limit, row_key_prefix=None, cursor=None):
        if cursor is not None:
            if self._fail_next_continuation:
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


def _start_partial_cycle(persistence, *, tenant: str = _TENANT_A) -> str:
    page = persistence.reconcile_list_indexes(
        tenant_id=tenant,
        minimum_projection_age=_MAINTENANCE_AGE,
        limit=1,
    )
    assert page.has_more is True
    assert page.next_cursor is not None
    return page.next_cursor


def _install_admission_gate(persistence):
    """Hold the first admitted continuation until the peer admission attempt finishes."""
    reconciler = persistence._list_index_reconciler
    original_admit = reconciler._admit_maintenance_page
    hold_winner = threading.Event()
    loser_may_try = threading.Event()
    release_winner = threading.Event()
    admission_calls = 0
    admission_lock = threading.Lock()

    def gated_admit(*, cycle_state) -> None:
        nonlocal admission_calls
        with admission_lock:
            admission_calls += 1
            call_number = admission_calls
        if call_number == 1:
            original_admit(cycle_state=cycle_state)
            hold_winner.set()
            loser_may_try.set()
            release_winner.wait(timeout=5)
            return
        original_admit(cycle_state=cycle_state)

    reconciler._admit_maintenance_page = gated_admit
    return hold_winner, loser_may_try, release_winner


def test_sf1_same_key_concurrent_continuation_one_admitted_one_rejected() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    _seed_orphans(store, tenant=_TENANT_A, partition_key=_PARTITION_A, count=4)
    cursor = _start_partial_cycle(persistence)
    hold_winner, loser_may_try, release_winner = _install_admission_gate(persistence)

    rejections: list[ProblemListIndexReconciliationError] = []
    winner_error: list[BaseException] = []

    def run_winner() -> None:
        try:
            persistence.reconcile_list_indexes(
                tenant_id=_TENANT_A,
                minimum_projection_age=_MAINTENANCE_AGE,
                limit=1,
                cursor=cursor,
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            winner_error.append(exc)

    def run_loser() -> None:
        assert loser_may_try.wait(timeout=5) is True
        try:
            persistence.reconcile_list_indexes(
                tenant_id=_TENANT_A,
                minimum_projection_age=_MAINTENANCE_AGE,
                limit=1,
                cursor=cursor,
            )
        except ProblemListIndexReconciliationError as exc:
            rejections.append(exc)

    winner = threading.Thread(target=run_winner)
    loser = threading.Thread(target=run_loser)
    winner.start()
    assert hold_winner.wait(timeout=5) is True
    loser.start()
    assert loser_may_try.wait(timeout=5) is True
    loser.join(timeout=5)
    assert len(rejections) == 1
    assert "page already in progress" in str(rejections[0])
    release_winner.set()
    winner.join(timeout=10)

    assert winner_error == []


def test_sf2_different_tenants_both_admitted() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    _seed_orphans(store, tenant=_TENANT_A, partition_key=_PARTITION_A, count=2)
    _seed_orphans(store, tenant=_TENANT_B, partition_key=_PARTITION_B, count=2)

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


def test_sf3_different_scopes_both_admitted() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    _seed_orphans(store, tenant=_TENANT_A, partition_key=_PARTITION_A, count=2)

    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def run_scope(scope: ProblemListScope) -> None:
        try:
            barrier.wait(timeout=5)
            persistence.reconcile_list_indexes(
                tenant_id=_TENANT_A,
                minimum_projection_age=_MAINTENANCE_AGE,
                scope=scope,
                limit=1,
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = (
        threading.Thread(target=run_scope, args=(ProblemListScope.OPEN,)),
        threading.Thread(target=run_scope, args=(ProblemListScope.RESOLVED,)),
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []


def test_sf4_winner_finds_issue_health_stays_degraded() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    _seed_orphans(store, tenant=_TENANT_A, partition_key=_PARTITION_A, count=3)
    cursor = _start_partial_cycle(persistence)
    hold_winner, loser_may_try, release_winner = _install_admission_gate(persistence)

    rejections: list[ProblemListIndexReconciliationError] = []

    def run_winner() -> None:
        persistence.reconcile_list_indexes(
            tenant_id=_TENANT_A,
            minimum_projection_age=_MAINTENANCE_AGE,
            limit=1,
            cursor=cursor,
        )

    def run_loser() -> None:
        assert loser_may_try.wait(timeout=5) is True
        try:
            persistence.reconcile_list_indexes(
                tenant_id=_TENANT_A,
                minimum_projection_age=_MAINTENANCE_AGE,
                limit=1,
                cursor=cursor,
            )
        except ProblemListIndexReconciliationError as exc:
            rejections.append(exc)

    winner = threading.Thread(target=run_winner)
    loser = threading.Thread(target=run_loser)
    winner.start()
    assert hold_winner.wait(timeout=5) is True
    loser.start()
    loser.join(timeout=5)
    assert len(rejections) == 1
    release_winner.set()
    winner.join(timeout=10)

    assert persistence.projection_health() is ProblemListProjectionHealth.DEGRADED


def test_sf5_exception_releases_ownership_retry_succeeds() -> None:
    store = _FailingQueryStore()
    persistence = _persistence_with_clock(store)
    _seed_orphans(store, tenant=_TENANT_A, partition_key=_PARTITION_A, count=2)
    cursor = _start_partial_cycle(persistence)
    store.arm_next_continuation_failure()

    with pytest.raises(RuntimeError, match="document store query failed"):
        persistence.reconcile_list_indexes(
            tenant_id=_TENANT_A,
            minimum_projection_age=_MAINTENANCE_AGE,
            limit=1,
            cursor=cursor,
        )

    page = persistence.reconcile_list_indexes(
        tenant_id=_TENANT_A,
        minimum_projection_age=_MAINTENANCE_AGE,
        limit=1,
        cursor=cursor,
    )
    assert page.examined >= 1


def test_sf6_previous_degraded_state_survives_failed_page() -> None:
    store = _FailingQueryStore()
    persistence = _persistence_with_clock(store)
    _seed_orphans(store, tenant=_TENANT_A, partition_key=_PARTITION_A, count=2)

    first = persistence.reconcile_list_indexes(
        tenant_id=_TENANT_A,
        minimum_projection_age=_MAINTENANCE_AGE,
        limit=1,
    )
    assert first.deleted >= 1
    assert persistence.projection_health() is ProblemListProjectionHealth.DEGRADED
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

    assert persistence.projection_health() is ProblemListProjectionHealth.DEGRADED


def test_sf7_completed_clean_cycle_prunes_registry() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    persistence.create(sample_problem(tenant_id=_TENANT_A, problem_id=mint_problem_id()))

    persistence.reconcile_list_indexes(
        tenant_id=_TENANT_A,
        minimum_projection_age=_MAINTENANCE_AGE,
        limit=100,
    )

    assert persistence.projection_health() is ProblemListProjectionHealth.HEALTHY
    assert _registry_size(persistence) == 0
