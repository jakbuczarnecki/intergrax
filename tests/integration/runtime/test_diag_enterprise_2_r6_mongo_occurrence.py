# © Artur Czarnecki. All rights reserved.

"""DIAG-ENTERPRISE-2-R6 real Mongo partition-atomic occurrence persistence qualification."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.integrations._shared.partition_atomic_conformance import (
    assert_partition_atomic_document_store_semantics,
)
from intergrax.runtime.diagnostics.persistence_conformance import (
    _sample_subject_ref,
    sample_occurrences,
    sample_problem,
)
from intergrax.runtime.diagnostics.problem_occurrence_aggregate_reconciliation import (
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
)

_TENANT = "diag-enterprise-2-r6-mongo"
_OBSERVED_AT = datetime(2026, 8, 31, 12, 30, tzinfo=UTC)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.external_proof,
    pytest.mark.network,
    pytest.mark.no_ci,
]


@pytest.fixture
def mongo_e2_r6_lifecycle(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
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


def test_mongo_partition_atomic_conformance(mongo_e2_r6_lifecycle: None) -> None:
    del mongo_e2_r6_lifecycle
    store = create_proof_document_store()
    try:
        assert_partition_atomic_document_store_semantics(store)
    finally:
        store.close()


def test_mongo_r6_concurrent_10k_occurrences_exact(mongo_e2_r6_lifecycle: None) -> None:
    del mongo_e2_r6_lifecycle
    store = create_proof_document_store()
    try:
        persistence = document_store_occurrence_persistence_for_tests(store)
        problem = sample_problem(tenant_id=_TENANT, observed_at=_OBSERVED_AT)
        barrier = threading.Barrier(worker_count)
        write_count = 10_500
        worker_count = 32

        def _append(index: int) -> None:
            subject = _sample_subject_ref(tenant_id=_TENANT)
            occurrence = sample_occurrences(
                subject_refs=(subject,),
                observed_at=_OBSERVED_AT + timedelta(microseconds=index),
            )[0]
            barrier.wait(timeout=30)
            persistence.append_if_absent(
                tenant_id=_TENANT,
                problem_id=problem.problem_id,
                occurrence=occurrence,
            )

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_append, index) for index in range(write_count)]
            for future in futures:
                future.result(timeout=120)

        scan = scan_occurrence_aggregate(
            persistence,
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
        )
        assert scan.occurrence_count == write_count

        duplicate = sample_occurrences(
            subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
            observed_at=_OBSERVED_AT,
        )[0]
        assert (
            persistence.append_if_absent(
                tenant_id=_TENANT,
                problem_id=problem.problem_id,
                occurrence=duplicate,
            )
            is ProblemOccurrenceAppendResult.CREATED
        )
        assert (
            persistence.append_if_absent(
                tenant_id=_TENANT,
                problem_id=problem.problem_id,
                occurrence=duplicate,
            )
            is ProblemOccurrenceAppendResult.ALREADY_EXISTS
        )
        scan_after_duplicate = scan_occurrence_aggregate(
            persistence,
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
        )
        assert scan_after_duplicate.occurrence_count == write_count + 1
    finally:
        store.close()
