# © Artur Czarnecki. All rights reserved.

"""DIAG-ENTERPRISE-2 bounded occurrence history proofs."""

from __future__ import annotations

from dataclasses import fields
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.runtime.diagnostics.persistence_conformance import (
    _sample_subject_ref,
    sample_occurrences,
    sample_problem,
)
from intergrax.runtime.diagnostics.problem_lifecycle import Problem
from intergrax.runtime.diagnostics.problem_occurrence_persistence import (
    ProblemOccurrenceAppendResult,
    ProblemOccurrencePersistenceIntegrityError,
)
from intergrax.runtime.diagnostics.problem_occurrence_query import (
    ProblemOccurrenceQueryCursorCodec,
    ProblemOccurrenceQueryCursorError,
)
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    TEST_PROBLEM_LIST_CURSOR_SECRET,
    document_store_lifecycle_stack_for_tests,
    document_store_occurrence_persistence_for_tests,
    document_store_problem_persistence_for_tests,
    in_memory_document_store_for_problem_tests,
    query_all_occurrences_for_problem,
)

pytestmark = pytest.mark.unit

_TENANT = "diag-enterprise-2"
_OBSERVED_AT = datetime(2026, 8, 26, 9, 0, tzinfo=UTC)


def test_problem_aggregate_has_no_unbounded_occurrence_fields() -> None:
    field_names = {field.name for field in fields(Problem)}
    assert "occurrences" not in field_names
    assert "current_subject_refs" not in field_names


def test_append_idempotent_and_paginated_history() -> None:
    store = in_memory_document_store_for_problem_tests()
    occurrence_persistence = document_store_occurrence_persistence_for_tests(store)
    subject_refs = (_sample_subject_ref(tenant_id=_TENANT),)
    occurrences = sample_occurrences(subject_refs=subject_refs, observed_at=_OBSERVED_AT)
    problem = sample_problem(tenant_id=_TENANT, subject_refs=subject_refs)

    first_result = occurrence_persistence.append_if_absent(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
        occurrence=occurrences[0],
    )
    assert first_result is ProblemOccurrenceAppendResult.CREATED
    for _ in range(99):
        assert (
            occurrence_persistence.append_if_absent(
                tenant_id=_TENANT,
                problem_id=problem.problem_id,
                occurrence=occurrences[0],
            )
            is ProblemOccurrenceAppendResult.ALREADY_EXISTS
        )

    page = occurrence_persistence.query_occurrences(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
        limit=10,
    )
    assert len(page.items) == 1
    stats = occurrence_persistence.aggregate_stats(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert stats is not None
    assert stats.occurrence_count == 1


def test_cursor_rejects_cross_problem_scope() -> None:
    store = in_memory_document_store_for_problem_tests()
    occurrence_persistence = document_store_occurrence_persistence_for_tests(store)
    problem_a = sample_problem(tenant_id=_TENANT)
    problem_b = sample_problem(tenant_id=_TENANT)
    for index, problem in enumerate((problem_a, problem_b)):
        subject = _sample_subject_ref(tenant_id=_TENANT)
        occurrence = sample_occurrences(
            subject_refs=(subject,),
            observed_at=_OBSERVED_AT + timedelta(seconds=index),
        )[0]
        occurrence_persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence,
        )

    forged_cursor = ProblemOccurrenceQueryCursorCodec(
        secret=TEST_PROBLEM_LIST_CURSOR_SECRET,
    ).encode(
        tenant_id=_TENANT,
        problem_id=problem_a.problem_id,
        store_cursor="forged",
    )
    with pytest.raises(ProblemOccurrenceQueryCursorError):
        occurrence_persistence.query_occurrences(
            tenant_id=_TENANT,
            problem_id=problem_b.problem_id,
            limit=1,
            cursor=forged_cursor,
        )


def test_100k_bounded_aggregate_and_paginated_history() -> None:
    store, problem_persistence, occurrence_persistence, lifecycle = (
        document_store_lifecycle_stack_for_tests()
    )
    del lifecycle, problem_persistence
    problem = sample_problem(tenant_id=_TENANT, occurrence_count=0)
    subject_base = _sample_subject_ref(tenant_id=_TENANT)

    for index in range(100_000):
        subject = _sample_subject_ref(tenant_id=_TENANT)
        occurrences = sample_occurrences(
            subject_refs=(subject,),
            observed_at=_OBSERVED_AT + timedelta(seconds=index),
        )
        occurrence_persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrences[0],
        )

    stats = occurrence_persistence.aggregate_stats(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert stats is not None
    assert stats.occurrence_count == 100_000

    bounded_page = occurrence_persistence.query_occurrences(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
        limit=100,
    )
    assert len(bounded_page.items) == 100
    assert bounded_page.has_more is True

    aggregate_field_count = len(tuple(fields(Problem)))
    bounded_problem = sample_problem(
        tenant_id=_TENANT,
        occurrence_count=100_000,
    )
    assert len(tuple(fields(bounded_problem))) == aggregate_field_count

    collected = query_all_occurrences_for_problem(
        occurrence_persistence,
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
        page_limit=500,
    )
    assert len(collected) == 100_000
    del subject_base


def test_malformed_occurrence_record_fails_closed() -> None:
    from intergrax.integrations.contracts.document_store import DocumentRecord
    from intergrax.runtime.diagnostics.document_store_problem_occurrence_persistence import (
        _occurrence_partition,
    )
    from intergrax.runtime.diagnostics.problem_occurrence_record_codec import (
        decode_problem_occurrence_record,
    )

    store = in_memory_document_store_for_problem_tests()
    occurrence_persistence = document_store_occurrence_persistence_for_tests(store)
    problem = sample_problem(tenant_id=_TENANT)
    partition = _occurrence_partition(_TENANT, problem.problem_id)
    store.put_if_absent(
        DocumentRecord(
            partition_key=partition,
            row_key="occ:0000000000000001:bad",
            data={"schema_version": "broken", "payload": {}},
        ),
    )
    with pytest.raises(ProblemOccurrencePersistenceIntegrityError):
        occurrence_persistence.query_occurrences(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            limit=10,
        )
    with pytest.raises(ProblemOccurrencePersistenceIntegrityError):
        decode_problem_occurrence_record({"schema_version": "broken", "payload": {}})


def test_migration_1000_legacy_occurrences_resumable_after_failure() -> None:
    from intergrax.integrations.contracts.document_store import DocumentRecord
    from intergrax.runtime.diagnostics.document_store_problem_persistence import (
        _document_partition,
        _record_row_key,
    )
    from intergrax.runtime.diagnostics.problem_occurrence_migration import (
        migrate_legacy_problem_inline_occurrences,
        verify_legacy_occurrences_migrated,
    )
    from intergrax.runtime.diagnostics.problem_record_codec import (
        _encode_legacy_problem_payload_v1,
    )

    store = in_memory_document_store_for_problem_tests()
    problem_persistence = document_store_problem_persistence_for_tests(store)
    occurrence_persistence = document_store_occurrence_persistence_for_tests(store)
    partition = _document_partition(_TENANT)
    problem = sample_problem(tenant_id=_TENANT, occurrence_count=1000)
    subject_refs = tuple(
        _sample_subject_ref(tenant_id=_TENANT) for _ in range(1000)
    )
    occurrences = sample_occurrences(
        subject_refs=subject_refs,
        observed_at=_OBSERVED_AT,
    )
    bounded = sample_problem(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
        subject_refs=(subject_refs[0],),
        occurrence_count=1000,
        observed_at=_OBSERVED_AT,
    )
    legacy_data = {
        "schema_version": "intergrax.diagnostic_problem.persistence.v1",
        "payload": _encode_legacy_problem_payload_v1(
            problem=bounded,
            current_subject_refs=subject_refs,
            occurrences=occurrences,
        ),
    }
    store.put_if_absent(
        DocumentRecord(
            partition_key=partition,
            row_key=_record_row_key(problem.problem_id),
            data=legacy_data,
        ),
    )

    first_page = migrate_legacy_problem_inline_occurrences(
        tenant_id=_TENANT,
        problem_persistence=problem_persistence,
        occurrence_persistence=occurrence_persistence,
        document_store=store,
        limit=500,
    )
    assert len(first_page.migrated_problem_ids) == 1
    assert first_page.has_more is False

    stats = occurrence_persistence.aggregate_stats(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert stats is not None
    assert stats.occurrence_count == 1000
    loaded = problem_persistence.get(tenant_id=_TENANT, problem_id=problem.problem_id)
    assert loaded is not None
    assert loaded.occurrence_count == 1000
    assert len(tuple(fields(loaded))) == 8
    assert verify_legacy_occurrences_migrated(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
        occurrence_persistence=occurrence_persistence,
        document_store=store,
    )
