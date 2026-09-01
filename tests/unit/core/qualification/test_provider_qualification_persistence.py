# © Artur Czarnecki. All rights reserved.

"""Durable provider qualification evidence persistence tests (PROVIDER-QUAL-3C)."""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone

import pytest

from intergrax.core.qualification import (
    ProviderQualificationEnvironmentMetadata,
    ProviderQualificationEvidenceKind,
    ProviderQualificationExecutor,
    ProviderQualificationResultSummary,
    ProviderQualificationRun,
    ProviderQualificationSubject,
    QualificationEvidence,
    QualificationRunId,
    QualificationStatus,
    new_qualification_run_id,
)
from intergrax.core.qualification.persistence import (
    DocumentStoreProviderQualificationPersistence,
    ProviderQualificationPersistenceConflictError,
    ProviderQualificationPersistenceIntegrityError,
    encode_provider_qualification_run,
    proof_receipt_to_provider_qualification_run,
    provider_qualification_run_to_proof_receipt,
    wire_provider_qualification_persistence,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.proofs.receipts.document_store import (
    proof_receipt_lookup_row_key,
    proof_receipt_partition_key,
    proof_receipt_to_document,
)
from intergrax.proofs.receipts.store import ProofReceiptStore

pytestmark = pytest.mark.unit

_EXECUTED_AT = datetime(2026, 8, 17, 12, 0, 0, tzinfo=timezone.utc)
_FIXED_RUN_ID = QualificationRunId("qual_run_0123456789abcdef0123456789abcdef")


def _subject() -> ProviderQualificationSubject:
    return ProviderQualificationSubject(
        provider_id="postgresql",
        provider_version="16.6",
        capability_id="collaborative_work.persistence.v1",
        domain="collaborative_work",
        intergrax_revision="bd657b431e2c020da0a89de45f6f3b448a48867a",
        qualification_suite_id="cw.postgresql.repository.v1",
        qualification_suite_version="1.0.0",
        environment_id="local-docker-qual-host",
        adapter_identity="intergrax.integrations.providers.relational_store.postgresql",
    )


def _run(
    *,
    run_id: QualificationRunId = _FIXED_RUN_ID,
    status: QualificationStatus = QualificationStatus.PRODUCTION_QUALIFIED,
    failed: int = 0,
) -> ProviderQualificationRun:
    return ProviderQualificationRun(
        qualification_run_id=run_id,
        subject=_subject(),
        status=status,
        executed_at=_EXECUTED_AT,
        executor=ProviderQualificationExecutor(
            executor_kind="local_cli",
            executor_id="qual-host-01",
            executor_version="2026.08.17",
        ),
        result_summary=ProviderQualificationResultSummary(
            passed=42,
            failed=failed,
            skipped=3,
            label="cw.postgresql.repository.v1",
        ),
        evidence=(
            QualificationEvidence(
                kind=ProviderQualificationEvidenceKind.SUITE_EXECUTION,
                code="suite.passed",
                ref="tests/integration/cw/test_postgresql_repository.py",
            ),
            QualificationEvidence(
                kind=ProviderQualificationEvidenceKind.LIVE_BACKEND,
                code="backend.live",
                label="postgresql-16.6",
            ),
        ),
        reproducibility="uv run pytest tests/integration/cw/test_postgresql_repository.py",
        limitations=("bounded local docker host",),
        source_revision="bd657b431e2c020da0a89de45f6f3b448a48867a",
        environment_metadata=ProviderQualificationEnvironmentMetadata(
            real_backend=True,
            mocks=False,
            sqlite_substitution=False,
            bounded_environment="docker-postgres-16",
        ),
    )


def test_persist_round_trip_and_lookup_by_qualification_run_id() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationPersistence(store)
    run = _run()

    persisted = persistence.persist(run)
    loaded = persistence.get_by_qualification_run_id(run.qualification_run_id)

    assert persisted == run
    assert loaded == run


def test_persist_is_idempotent_for_identical_run() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationPersistence(store)
    run = _run()

    first = persistence.persist(run)
    second = persistence.persist(run)

    assert first == run
    assert second == run
    assert persistence.get_by_qualification_run_id(run.qualification_run_id) == run


def test_persist_conflicting_same_run_id_fails_closed() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationPersistence(store)
    run = _run()
    conflicting = _run(failed=1)

    persistence.persist(run)

    with pytest.raises(
        ProviderQualificationPersistenceConflictError,
        match="conflicting provider qualification run",
    ):
        persistence.persist(conflicting)


def test_unknown_qualification_run_id_returns_none() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationPersistence(store)

    assert persistence.get_by_qualification_run_id(new_qualification_run_id()) is None


def test_corrupt_persisted_record_fails_closed() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationPersistence(store)
    run = _run()
    receipt = provider_qualification_run_to_proof_receipt(run)
    corrupt = proof_receipt_to_document(
        receipt.model_copy(update={"domain_evidence": {"schema_version": "broken"}}),
    )
    store.put(corrupt)

    with pytest.raises(ProviderQualificationPersistenceIntegrityError):
        persistence.get_by_qualification_run_id(run.qualification_run_id)


def test_restart_survives_new_persistence_adapter_instance() -> None:
    store = InMemoryDocumentStore()
    run = _run()

    first = DocumentStoreProviderQualificationPersistence(store)
    first.persist(run)
    first.close()

    second = DocumentStoreProviderQualificationPersistence(store)
    try:
        loaded = second.get_by_qualification_run_id(run.qualification_run_id)
        assert loaded == run
    finally:
        second.close()


def test_qualification_run_id_remains_authoritative_and_proof_id_differs() -> None:
    run = _run()
    receipt = provider_qualification_run_to_proof_receipt(run)

    assert receipt.run_id == str(run.qualification_run_id)
    assert receipt.proof_id != str(run.qualification_run_id)
    assert receipt.metadata["qualification_run_id"] == str(run.qualification_run_id)

    restored = proof_receipt_to_provider_qualification_run(receipt)
    assert restored.qualification_run_id == run.qualification_run_id


def test_persisted_projection_contains_no_secret_keys() -> None:
    run = _run()
    receipt = provider_qualification_run_to_proof_receipt(run)
    serialized = json.dumps(receipt.model_dump(mode="json")).lower()

    for forbidden in ("password", "token", "secret", "api_key", "credentials"):
        assert forbidden not in serialized


def test_encode_decode_provider_qualification_run_round_trip() -> None:
    run = _run()
    encoded = encode_provider_qualification_run(run)
    receipt = provider_qualification_run_to_proof_receipt(run)

    restored = proof_receipt_to_provider_qualification_run(receipt)
    assert restored == run
    assert encoded["qualification_run_id"] == str(run.qualification_run_id)


def test_wire_uses_existing_proof_receipt_document_store_path() -> None:
    store = InMemoryDocumentStore()
    persistence = wire_provider_qualification_persistence(document_store=store)
    run = _run()

    persistence.persist(run)
    proof_store = ProofReceiptStore(store)
    loaded_receipt = proof_store.get(
        "intergrax.provider_qualification",
        "provider_qualification",
        str(run.qualification_run_id),
    )

    assert loaded_receipt is not None
    assert proof_receipt_to_provider_qualification_run(loaded_receipt) == run
    assert proof_receipt_partition_key("intergrax.provider_qualification").startswith(
        "proof_receipts/",
    )


def test_platform_reuse_assertion_no_parallel_qualification_storage_backend() -> None:
    import intergrax.core.qualification.persistence as persistence_module

    source = inspect.getsource(persistence_module)
    assert "ProofReceipt" in source
    assert "proof_receipt_to_document" in source
    assert "ProviderQualificationDatabase" not in source
    assert "ProviderQualificationStore" not in source
    assert "ProviderQualificationEvidenceStore" not in source


def test_malformed_proof_receipt_schema_fails_closed() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationPersistence(store)
    partition_key = proof_receipt_partition_key("intergrax.provider_qualification")
    row_key = proof_receipt_lookup_row_key(
        "provider_qualification",
        str(_FIXED_RUN_ID),
    )
    store.put(
        DocumentRecord(
            partition_key=partition_key,
            row_key=row_key,
            data={"schema_version": "intergrax.proof_receipt.v9"},
        ),
    )

    with pytest.raises(ProviderQualificationPersistenceIntegrityError):
        persistence.get_by_qualification_run_id(_FIXED_RUN_ID)
