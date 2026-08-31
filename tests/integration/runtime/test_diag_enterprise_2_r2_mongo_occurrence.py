# © Artur Czarnecki. All rights reserved.

"""DIAG-ENTERPRISE-2-R2 real Mongo occurrence persistence qualification."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.runtime.diagnostics.persistence_conformance import (
    _sample_subject_ref,
    sample_occurrences,
    sample_problem,
)
from intergrax.runtime.diagnostics.problem_occurrence_persistence import (
    ProblemOccurrenceAppendResult,
)
from intergrax.runtime.diagnostics.problem_occurrence_query import (
    ProblemOccurrenceQueryCursorCodec,
    ProblemOccurrenceQueryCursorError,
)
from tests.integration.applications.architecture.harden_4f_mongo_support import (
    cleanup_proof_tenant,
    create_proof_document_store,
    ensure_mongo_running,
    require_docker_for_harden_4f_proof,
)
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    TEST_PROBLEM_LIST_CURSOR_SECRET,
    document_store_occurrence_persistence_for_tests,
    document_store_problem_persistence_for_tests,
)

_TENANT = "diag-enterprise-2-r2-mongo"
_OBSERVED_AT = datetime(2026, 8, 31, 9, 0, tzinfo=UTC)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.external_proof,
    pytest.mark.network,
    pytest.mark.no_ci,
]


@pytest.fixture
def mongo_e2_r2_lifecycle(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
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


def test_mongo_occurrence_persistence_matrix(mongo_e2_r2_lifecycle: None) -> None:
    del mongo_e2_r2_lifecycle
    store = create_proof_document_store()
    try:
        persistence = document_store_occurrence_persistence_for_tests(store)
        problem_persistence = document_store_problem_persistence_for_tests(store)
        problem = sample_problem(tenant_id=_TENANT, observed_at=_OBSERVED_AT)

        # M1 — append unique occurrences
        subjects = tuple(_sample_subject_ref(tenant_id=_TENANT) for _ in range(3))
        occurrences = sample_occurrences(
            subject_refs=subjects,
            observed_at=_OBSERVED_AT,
        )
        for occurrence in occurrences:
            assert (
                persistence.append_if_absent(
                    tenant_id=_TENANT,
                    problem_id=problem.problem_id,
                    occurrence=occurrence,
                )
                is ProblemOccurrenceAppendResult.CREATED
            )

        # M2 — duplicate retry
        assert (
            persistence.append_if_absent(
                tenant_id=_TENANT,
                problem_id=problem.problem_id,
                occurrence=occurrences[0],
            )
            is ProblemOccurrenceAppendResult.ALREADY_EXISTS
        )

        # M3 — pagination
        page = persistence.query_occurrences(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            limit=2,
        )
        assert len(page.items) == 2
        assert page.has_more is True
        page2 = persistence.query_occurrences(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            limit=2,
            cursor=page.next_cursor,
        )
        assert len(page2.items) == 1

        # M4 — wrong tenant/problem cursor reject
        other_problem = sample_problem(tenant_id=_TENANT)
        forged = ProblemOccurrenceQueryCursorCodec(
            secret=TEST_PROBLEM_LIST_CURSOR_SECRET,
        ).encode(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            store_cursor="forged",
        )
        with pytest.raises(ProblemOccurrenceQueryCursorError):
            persistence.query_occurrences(
                tenant_id=_TENANT,
                problem_id=other_problem.problem_id,
                limit=1,
                cursor=forged,
            )

        stats = persistence.aggregate_stats(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
        )
        assert stats is not None
        assert stats.occurrence_count == 3

        # M5 — restart/recreate persistence instance
        restarted = document_store_occurrence_persistence_for_tests(store)
        stats_restarted = restarted.aggregate_stats(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
        )
        assert stats_restarted == stats

        # M6 — S1 crash recovery via retry
        crash_subject = _sample_subject_ref(tenant_id=_TENANT)
        crash_occurrence = sample_occurrences(
            subject_refs=(crash_subject,),
            observed_at=_OBSERVED_AT + timedelta(hours=1),
        )[0]
        from intergrax.runtime.diagnostics.document_store_problem_occurrence_persistence import (
            _occurrence_partition,
            _occurrence_row_key,
        )
        from intergrax.runtime.diagnostics.problem_occurrence_id import (
            problem_occurrence_id_for,
        )
        from intergrax.runtime.diagnostics.problem_occurrence_record_codec import (
            encode_problem_occurrence_record,
        )
        from intergrax.integrations.contracts.document_store import DocumentRecord

        crash_id = problem_occurrence_id_for(crash_occurrence)
        partition = _occurrence_partition(_TENANT, problem.problem_id)
        store.put_if_absent(
            DocumentRecord(
                partition_key=partition,
                row_key=_occurrence_row_key(crash_occurrence, occurrence_id=crash_id),
                data=encode_problem_occurrence_record(crash_occurrence),
            ),
        )
        assert (
            restarted.append_if_absent(
                tenant_id=_TENANT,
                problem_id=problem.problem_id,
                occurrence=crash_occurrence,
            )
            is ProblemOccurrenceAppendResult.ALREADY_EXISTS
        )
        stats_after_s1 = restarted.aggregate_stats(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
        )
        assert stats_after_s1 is not None
        assert stats_after_s1.occurrence_count == 4

        # M7 — concurrent duplicate
        dup_subject = _sample_subject_ref(tenant_id=_TENANT)
        dup_occurrence = sample_occurrences(
            subject_refs=(dup_subject,),
            observed_at=_OBSERVED_AT + timedelta(hours=2),
        )[0]
        barrier = threading.Barrier(2)
        dup_results: list[ProblemOccurrenceAppendResult] = []

        def _append_dup() -> None:
            barrier.wait(timeout=10)
            dup_results.append(
                restarted.append_if_absent(
                    tenant_id=_TENANT,
                    problem_id=problem.problem_id,
                    occurrence=dup_occurrence,
                ),
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_append_dup), executor.submit(_append_dup)]
            for future in futures:
                future.result(timeout=30)
        assert sorted(dup_results, key=str) == sorted(
            [
                ProblemOccurrenceAppendResult.CREATED,
                ProblemOccurrenceAppendResult.ALREADY_EXISTS,
            ],
            key=str,
        )

        # M8 — concurrent distinct occurrences
        distinct_a = sample_occurrences(
            subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
            observed_at=_OBSERVED_AT + timedelta(hours=3),
        )[0]
        distinct_b = sample_occurrences(
            subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
            observed_at=_OBSERVED_AT + timedelta(hours=4),
        )[0]
        barrier2 = threading.Barrier(2)
        distinct_results: list[ProblemOccurrenceAppendResult] = []

        def _append_distinct(occurrence) -> None:
            barrier2.wait(timeout=10)
            distinct_results.append(
                restarted.append_if_absent(
                    tenant_id=_TENANT,
                    problem_id=problem.problem_id,
                    occurrence=occurrence,
                ),
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(_append_distinct, distinct_a),
                executor.submit(_append_distinct, distinct_b),
            ]
            for future in futures:
                future.result(timeout=30)
        assert all(
            result is ProblemOccurrenceAppendResult.CREATED
            for result in distinct_results
        )

        final_stats = restarted.aggregate_stats(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
        )
        assert final_stats is not None
        assert final_stats.occurrence_count == 7

        # M9 — Problem aggregate convergence
        agg_problem = sample_problem(
            tenant_id=_TENANT,
            occurrence_count=0,
            observed_at=_OBSERVED_AT,
        )
        agg_subject = _sample_subject_ref(tenant_id=_TENANT)
        agg_occurrence = sample_occurrences(
            subject_refs=(agg_subject,),
            observed_at=_OBSERVED_AT + timedelta(days=1),
        )[0]
        problem_persistence.create(agg_problem, indexed_subject_refs=(agg_subject,))
        restarted.append_if_absent(
            tenant_id=_TENANT,
            problem_id=agg_problem.problem_id,
            occurrence=agg_occurrence,
        )
        agg_stats = restarted.aggregate_stats(
            tenant_id=_TENANT,
            problem_id=agg_problem.problem_id,
        )
        assert agg_stats is not None
        assert agg_stats.occurrence_count == 1
        from intergrax.runtime.diagnostics.problem_occurrence_aggregate_convergence import (
            converge_problem_from_durable_stats,
        )

        loaded = problem_persistence.get(
            tenant_id=_TENANT,
            problem_id=agg_problem.problem_id,
        )
        assert loaded is not None
        converged = converge_problem_from_durable_stats(
            loaded,
            stats=agg_stats,
        )
        updated = problem_persistence.update(
            converged,
            expected_version=loaded.record_version,
        )
        assert updated.occurrence_count == 1
        assert updated.first_seen_at == agg_occurrence.observed_at
        assert updated.last_seen_at == agg_occurrence.observed_at
    finally:
        store.close()
