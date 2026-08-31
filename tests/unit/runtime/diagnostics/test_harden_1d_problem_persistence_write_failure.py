# © Artur Czarnecki. All rights reserved.

"""HARDEN-1D — DocumentStoreProblemPersistence must not report false write success."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.runtime.diagnostics.document_store_problem_persistence import (
    DocumentStoreProblemPersistence,
)
from intergrax.runtime.diagnostics.persistence_conformance import (
    _sample_subject_ref,
    query_all_problems_for_tenant,
    sample_problem,
    sample_subject_refs,
)
from intergrax.runtime.diagnostics.problem_lifecycle import Problem
from testing_support.delegating_failing_conditional_document_store import (
    ControlledDocumentStoreWriteFailure,
    DelegatingFailingConditionalDocumentStore,
    DocumentStoreWriteFailureMode,
)
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    document_store_problem_persistence_for_tests,
)

pytestmark = pytest.mark.unit

_OBSERVED_AT_LATER = datetime(2026, 8, 29, 10, 0, tzinfo=UTC)


def test_harden_1d_create_write_failure_is_not_reported_as_success() -> None:
    store = DelegatingFailingConditionalDocumentStore()
    store.set_write_failure_mode(DocumentStoreWriteFailureMode.FAIL_WRITES)
    persistence = document_store_problem_persistence_for_tests(store)
    record = sample_problem(tenant_id="tenant-harden-1d-create")

    with pytest.raises(ControlledDocumentStoreWriteFailure):
        persistence.create(record)

    assert persistence.get(tenant_id=record.tenant_id, problem_id=record.problem_id) is None
    assert query_all_problems_for_tenant(persistence, record.tenant_id) == ()


def test_harden_1d_update_cas_write_failure_is_not_reported_as_success() -> None:
    store = DelegatingFailingConditionalDocumentStore()
    persistence = document_store_problem_persistence_for_tests(store)
    created = sample_problem(tenant_id="tenant-harden-1d-update")
    assert persistence.create(created) == created

    subject_a = sample_subject_refs(created)[0]
    subject_b = _sample_subject_ref(tenant_id=created.tenant_id)
    updated = Problem(
        problem_id=created.problem_id,
        tenant_id=created.tenant_id,
        status=created.status,
        first_seen_at=created.first_seen_at,
        last_seen_at=_OBSERVED_AT_LATER,
        occurrence_count=created.occurrence_count + 1,
        provenance=created.provenance,
        record_version=2,
    )

    store.set_write_failure_mode(DocumentStoreWriteFailureMode.FAIL_WRITES)
    with pytest.raises(ControlledDocumentStoreWriteFailure):
        persistence.update(updated, expected_version=1, indexed_subject_refs=(subject_b,))

    stored = persistence.get(tenant_id=created.tenant_id, problem_id=created.problem_id)
    assert stored == created
    assert (
        persistence.find_by_subject_ref(
            tenant_id=created.tenant_id,
            subject_ref=subject_b,
        )
        is None
    )


def test_harden_1d_store_recovery_allows_subsequent_create_and_update() -> None:
    store = DelegatingFailingConditionalDocumentStore()
    persistence = document_store_problem_persistence_for_tests(store)
    record = sample_problem(tenant_id="tenant-harden-1d-recovery")

    store.set_write_failure_mode(DocumentStoreWriteFailureMode.FAIL_WRITES)
    with pytest.raises(ControlledDocumentStoreWriteFailure):
        persistence.create(record)

    store.set_write_failure_mode(DocumentStoreWriteFailureMode.HEALTHY)
    assert persistence.create(record) == record

    subject_a = sample_subject_refs(record)[0]
    subject_b = _sample_subject_ref(tenant_id=record.tenant_id)
    updated = Problem(
        problem_id=record.problem_id,
        tenant_id=record.tenant_id,
        status=record.status,
        first_seen_at=record.first_seen_at,
        last_seen_at=_OBSERVED_AT_LATER,
        occurrence_count=record.occurrence_count + 1,
        provenance=record.provenance,
        record_version=2,
    )

    store.set_write_failure_mode(DocumentStoreWriteFailureMode.FAIL_WRITES)
    with pytest.raises(ControlledDocumentStoreWriteFailure):
        persistence.update(updated, expected_version=1, indexed_subject_refs=(subject_b,))

    store.set_write_failure_mode(DocumentStoreWriteFailureMode.HEALTHY)
    assert (
        persistence.update(
            updated,
            expected_version=1,
            indexed_subject_refs=(subject_b,),
        )
        == updated
    )
