# © Artur Czarnecki. All rights reserved.

"""DIAG-ENTERPRISE-2-R5 real Mongo snapshot-safe aggregate reconciliation qualification."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.runtime.diagnostics.persistence_conformance import (
    _sample_subject_ref,
    sample_occurrences,
    sample_problem,
)
from intergrax.runtime.diagnostics.problem_lifecycle import (
    ProblemOccurrenceAggregateHealth,
)
from intergrax.runtime.diagnostics.problem_occurrence_aggregate_reconciliation import (
    mark_problem_reconciliation_required,
    reconcile_problem_occurrence_aggregate,
    scan_occurrence_aggregate,
)
from tests.integration.applications.architecture.harden_4f_mongo_support import (
    cleanup_proof_tenant,
    create_proof_document_store,
    ensure_mongo_running,
    require_docker_for_harden_4f_proof,
)
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    document_store_occurrence_persistence_for_tests,
    document_store_problem_persistence_for_tests,
)
from tests.unit.runtime.diagnostics.test_diag_enterprise_2_r5_aggregate_reconciliation import (
    _LateInsertDuringScanPersistence,
    _seed_n_occurrences,
)

_TENANT = "diag-enterprise-2-r5-mongo"
_OBSERVED_AT = datetime(2026, 8, 31, 10, 30, tzinfo=UTC)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.external_proof,
    pytest.mark.network,
    pytest.mark.no_ci,
]


@pytest.fixture
def mongo_e2_r5_lifecycle(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from tests.integration.applications.architecture.harden_4f_mongo_support import proof_env

    env = proof_env()
    monkeypatch.setenv("INTERGRAX_MONGODB_URI", env["INTERGRAX_MONGODB_URI"])
    monkeypatch.setenv("INTERGRAX_MONGODB_DATABASE", env["INTERGRAX_MONGODB_DATABASE"])
    monkeypatch.setenv("INTERGRAX_MONGODB_COLLECTION", env["INTERGRAX_MONGODB_COLLECTION"])
    require_docker_for_harden_4f_proof()
    ensure_mongo_running()
    cleanup_proof_tenant(tenant_id=_TENANT)
    try:
        yield
    finally:
        cleanup_proof_tenant(tenant_id=_TENANT)
        ensure_mongo_running()


def test_mongo_r5_late_insert_repair_converges(mongo_e2_r5_lifecycle: None) -> None:
    del mongo_e2_r5_lifecycle
    store = create_proof_document_store()
    try:
        base_persistence = document_store_occurrence_persistence_for_tests(store)
        problem_persistence = document_store_problem_persistence_for_tests(store)
        problem = sample_problem(tenant_id=_TENANT, occurrence_count=0, observed_at=_OBSERVED_AT)
        problem_persistence.create(problem, indexed_subject_refs=())
        _seed_n_occurrences(base_persistence, problem=problem, count=10_500)

        stale = mark_problem_reconciliation_required(problem)
        problem_persistence.update(stale, expected_version=1)
        late = sample_occurrences(
            subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
            observed_at=_OBSERVED_AT + timedelta(seconds=10_499, microseconds=500000),
        )[0]
        intercepting = _LateInsertDuringScanPersistence(
            base_persistence,
            trigger_after_items=500,
            late_occurrence=late,
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
        )
        repaired = reconcile_problem_occurrence_aggregate(
            stale,
            occurrence_persistence=intercepting,
            problem_persistence=problem_persistence,
            page_size=500,
        )
        assert repaired.occurrence_count == 10_501
        assert repaired.occurrence_aggregate_health is ProblemOccurrenceAggregateHealth.CONSISTENT

        second = reconcile_problem_occurrence_aggregate(
            repaired,
            occurrence_persistence=intercepting,
            problem_persistence=problem_persistence,
            page_size=500,
        )
        assert second.occurrence_count == 10_501
        scan = scan_occurrence_aggregate(
            intercepting,
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
        )
        assert scan.occurrence_count == 10_501
    finally:
        store.close()
