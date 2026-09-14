# © Artur Czarnecki. All rights reserved.

"""NPSC-5F DocumentStore v1/v2 persistence conflict hardening."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.npsc5f_compatibility import (
    AmbiguousLegacyExecutionIdentityError,
    BackgroundExecutionIdentityConflictError,
    LegacyBackgroundExecutionIdentityIncompatibleError,
)
from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.runtime.background_execution.identity_dual_read import (
    reconcile_dual_read_records,
)
from intergrax.runtime.background_execution.identity_persistence import (
    DocumentStoreBackgroundExecutionIdentityPersistence,
)
from intergrax.runtime.background_execution.identity_record_codec import (
    BG_EXEC_IDENTITY_DOCUMENT_PARTITION_V1,
    BG_EXEC_IDENTITY_DOCUMENT_PARTITION_V2,
    DecodedBackgroundIdentityRecord,
    InvalidBackgroundExecutionIdentityV1RecordError,
    InvalidBackgroundExecutionIdentityV2RecordError,
    decode_document_identity_v1_record,
    decode_document_identity_v2_record,
    same_identity_quadruplet,
)
from intergrax.runtime.background_execution.transport_ref import (
    BackgroundTransportExecutionRef,
)
from intergrax.runtime.observability.causal_evidence_enrichment import (
    CanonicalExecutionIdLookupPort,
    CanonicalExecutionIdLookupResult,
)
from tests.unit.runtime.observability.test_npsc5f_v1_v2_compatibility import (
    _TenantScopedLookup,
)

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"


def _transport(
    transport_task_id: str = "transport-1",
) -> BackgroundTransportExecutionRef:
    return BackgroundTransportExecutionRef(
        tenant_id=_TENANT,
        provider="celery",
        transport_task_id=transport_task_id,
    )


def _row_key(transport: BackgroundTransportExecutionRef) -> str:
    return f"{transport.provider}:{transport.transport_task_id}"


def _put_v2(
    store: InMemoryDocumentStore,
    *,
    row_key: str,
    task_id: object,
    run_id: object,
    attempt_id: object,
    execution_id: object | None,
) -> None:
    data: dict[str, object] = {
        "task_id": str(task_id),
        "run_id": str(run_id),
        "attempt_id": str(attempt_id),
    }
    if execution_id is not None:
        data["execution_id"] = str(execution_id)
    store.put_if_absent(
        DocumentRecord(
            partition_key=f"{BG_EXEC_IDENTITY_DOCUMENT_PARTITION_V2}:{_TENANT}",
            row_key=row_key,
            data=data,
        )
    )


def _put_v1(
    store: InMemoryDocumentStore,
    *,
    row_key: str,
    task_id: object,
    run_id: object,
    attempt_id: object,
) -> None:
    store.put_if_absent(
        DocumentRecord(
            partition_key=f"{BG_EXEC_IDENTITY_DOCUMENT_PARTITION_V1}:{_TENANT}",
            row_key=row_key,
            data={
                "task_id": str(task_id),
                "run_id": str(run_id),
                "attempt_id": str(attempt_id),
            },
        )
    )


def test_v2_partition_valid_complete_shape_passes() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreBackgroundExecutionIdentityPersistence(store)
    transport = _transport()
    execution_id = mint_execution_id()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    _put_v2(
        store,
        row_key=_row_key(transport),
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    loaded = persistence.load(transport)
    assert loaded is not None
    assert loaded.execution_id == execution_id


def test_v2_partition_legacy_three_field_shape_fail_closed() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreBackgroundExecutionIdentityPersistence(store)
    transport = _transport("legacy-in-v2")
    _put_v2(
        store,
        row_key=_row_key(transport),
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=None,
    )
    with pytest.raises(InvalidBackgroundExecutionIdentityV2RecordError):
        persistence.load(transport)


def test_v1_partition_valid_legacy_shape_uses_legacy_path() -> None:
    store = InMemoryDocumentStore()
    execution_id = mint_execution_id()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    lookup = _TenantScopedLookup(by_tenant={_TENANT: execution_id})
    persistence = DocumentStoreBackgroundExecutionIdentityPersistence(
        store,
        legacy_lookup=lookup,
    )
    transport = _transport("v1-only")
    _put_v1(
        store,
        row_key=_row_key(transport),
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    loaded = persistence.load(transport)
    assert loaded is not None
    assert loaded.execution_id == execution_id


def test_v1_partition_v2_shape_fail_closed() -> None:
    with pytest.raises(InvalidBackgroundExecutionIdentityV1RecordError):
        decode_document_identity_v1_record(
            {
                "task_id": str(mint_task_id()),
                "run_id": str(mint_run_id()),
                "attempt_id": str(mint_attempt_id()),
                "execution_id": str(mint_execution_id()),
            }
        )


def test_matching_v1_and_v2_partitions_pass() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreBackgroundExecutionIdentityPersistence(store)
    transport = _transport("aligned")
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    row = _row_key(transport)
    _put_v2(
        store,
        row_key=row,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    _put_v1(
        store,
        row_key=row,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    loaded = persistence.load(transport)
    assert loaded is not None
    assert loaded.execution_id == execution_id


def test_same_task_different_run_fail_closed() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreBackgroundExecutionIdentityPersistence(store)
    transport = _transport("run-mismatch")
    task_id = mint_task_id()
    attempt_id = mint_attempt_id()
    row = _row_key(transport)
    _put_v2(
        store,
        row_key=row,
        task_id=task_id,
        run_id=mint_run_id(),
        attempt_id=attempt_id,
        execution_id=mint_execution_id(),
    )
    _put_v1(
        store,
        row_key=row,
        task_id=task_id,
        run_id=mint_run_id(),
        attempt_id=attempt_id,
    )
    with pytest.raises(BackgroundExecutionIdentityConflictError):
        persistence.load(transport)


def test_same_task_run_different_attempt_fail_closed() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreBackgroundExecutionIdentityPersistence(store)
    transport = _transport("attempt-mismatch")
    task_id = mint_task_id()
    run_id = mint_run_id()
    row = _row_key(transport)
    _put_v2(
        store,
        row_key=row,
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    _put_v1(
        store,
        row_key=row,
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
    )
    with pytest.raises(BackgroundExecutionIdentityConflictError):
        persistence.load(transport)


def test_different_task_fail_closed() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreBackgroundExecutionIdentityPersistence(store)
    transport = _transport("task-mismatch")
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    row = _row_key(transport)
    _put_v2(
        store,
        row_key=row,
        task_id=mint_task_id(),
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=mint_execution_id(),
    )
    _put_v1(
        store,
        row_key=row,
        task_id=mint_task_id(),
        run_id=run_id,
        attempt_id=attempt_id,
    )
    with pytest.raises(BackgroundExecutionIdentityConflictError):
        persistence.load(transport)


def test_v2_quadruplet_differs_when_execution_id_differs() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_a = mint_execution_id()
    execution_b = mint_execution_id()
    left = DecodedBackgroundIdentityRecord(
        kind="complete_v2",
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_a,
    )
    right = DecodedBackgroundIdentityRecord(
        kind="complete_v2",
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_a,
    )
    assert same_identity_quadruplet(left, right) is True
    right_other = DecodedBackgroundIdentityRecord(
        kind="complete_v2",
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_b,
    )
    assert same_identity_quadruplet(left, right_other) is False


def test_regression_v2_legacy_shape_plus_v1_same_task_different_run() -> None:
    """Audit bug: v2-partition legacy record was misclassified as v1."""
    store = InMemoryDocumentStore()
    persistence = DocumentStoreBackgroundExecutionIdentityPersistence(store)
    transport = _transport("audit-run")
    task_id = mint_task_id()
    row = _row_key(transport)
    _put_v2(
        store,
        row_key=row,
        task_id=task_id,
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=None,
    )
    _put_v1(
        store,
        row_key=row,
        task_id=task_id,
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )
    with pytest.raises(InvalidBackgroundExecutionIdentityV2RecordError):
        persistence.load(transport)


def test_regression_v2_complete_plus_v1_same_task_run_different_attempt() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreBackgroundExecutionIdentityPersistence(store)
    transport = _transport("audit-attempt")
    task_id = mint_task_id()
    run_id = mint_run_id()
    row = _row_key(transport)
    _put_v2(
        store,
        row_key=row,
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    _put_v1(
        store,
        row_key=row,
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
    )
    with pytest.raises(BackgroundExecutionIdentityConflictError):
        persistence.load(transport)


def test_v1_only_unresolved_lookup_still_fail_closed() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreBackgroundExecutionIdentityPersistence(store)
    transport = _transport("unresolved")
    _put_v1(
        store,
        row_key=_row_key(transport),
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )
    with pytest.raises(LegacyBackgroundExecutionIdentityIncompatibleError):
        persistence.load(transport)


class _AmbiguousLookup(CanonicalExecutionIdLookupPort):
    def lookup_execution_id(self, **kwargs: object) -> CanonicalExecutionIdLookupResult:
        _ = kwargs
        return CanonicalExecutionIdLookupResult(outcome="ambiguous")


def test_v1_only_ambiguous_lookup_still_fail_closed() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreBackgroundExecutionIdentityPersistence(
        store,
        legacy_lookup=_AmbiguousLookup(),
    )
    transport = _transport("ambiguous")
    _put_v1(
        store,
        row_key=_row_key(transport),
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )
    with pytest.raises(AmbiguousLegacyExecutionIdentityError):
        persistence.load(transport)


def test_decode_document_identity_v2_record_round_trip() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    decoded = decode_document_identity_v2_record(
        {
            "task_id": str(task_id),
            "run_id": str(run_id),
            "attempt_id": str(attempt_id),
            "execution_id": str(execution_id),
        }
    )
    assert decoded.kind == "complete_v2"
    outcome = reconcile_dual_read_records(
        v2_candidate=decoded,
        v1_candidate=None,
        tenant_id=_TENANT,
        lookup=None,
    )
    assert outcome.execution_id == execution_id
