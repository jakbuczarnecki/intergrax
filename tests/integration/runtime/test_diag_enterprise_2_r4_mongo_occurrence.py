# © Artur Czarnecki. All rights reserved.

"""DIAG-ENTERPRISE-2-R4 real Mongo aggregate reconciliation qualification."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.runtime.diagnostics.persistence_conformance import (
    _sample_subject_ref,
    sample_occurrences,
    sample_problem,
)
from intergrax.runtime.diagnostics.problem_occurrence_aggregate_reconciliation import (
    ProblemOccurrenceAggregateHealth,
    mark_problem_reconciliation_required,
    reconcile_problem_occurrence_aggregate,
    scan_occurrence_aggregate,
)
from intergrax.runtime.diagnostics.problem_occurrence_persistence import (
    ProblemOccurrenceAppendResult,
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

_TENANT = "diag-enterprise-2-r4-mongo"
_OBSERVED_AT = datetime(2026, 8, 31, 9, 0, tzinfo=UTC)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.external_proof,
    pytest.mark.network,
    pytest.mark.no_ci,
]


@pytest.fixture
def mongo_e2_r4_lifecycle(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
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


def test_mongo_r4_repair_after_injected_aggregate_failure(mongo_e2_r4_lifecycle: None) -> None:
    del mongo_e2_r4_lifecycle
    store = create_proof_document_store()
    try:
        occurrence_persistence = document_store_occurrence_persistence_for_tests(store)
        problem_persistence = document_store_problem_persistence_for_tests(store)
        problem = sample_problem(tenant_id=_TENANT, occurrence_count=1, observed_at=_OBSERVED_AT)
        subject = _sample_subject_ref(tenant_id=_TENANT)
        occurrence = sample_occurrences(subject_refs=(subject,), observed_at=_OBSERVED_AT)[0]
        problem_persistence.create(problem, indexed_subject_refs=(subject,))
        occurrence_persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence,
        )

        loaded = problem_persistence.get(tenant_id=_TENANT, problem_id=problem.problem_id)
        assert loaded is not None
        stale = mark_problem_reconciliation_required(loaded)
        problem_persistence.update(stale, expected_version=loaded.record_version)

        extra = sample_occurrences(
            subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
            observed_at=_OBSERVED_AT + timedelta(minutes=5),
        )[0]
        occurrence_persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=extra,
        )

        restarted = document_store_occurrence_persistence_for_tests(store)
        scan_before = scan_occurrence_aggregate(
            restarted,
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
        )
        assert scan_before.occurrence_count == 2

        repaired = reconcile_problem_occurrence_aggregate(
            stale,
            occurrence_persistence=restarted,
            problem_persistence=problem_persistence,
            page_size=500,
        )
        assert repaired.occurrence_count == 2
        assert repaired.occurrence_aggregate_health is ProblemOccurrenceAggregateHealth.CONSISTENT

        assert (
            restarted.append_if_absent(
                tenant_id=_TENANT,
                problem_id=problem.problem_id,
                occurrence=extra,
            )
            is ProblemOccurrenceAppendResult.ALREADY_EXISTS
        )
        final_scan = scan_occurrence_aggregate(
            restarted,
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
        )
        assert final_scan.occurrence_count == 2
    finally:
        store.close()
