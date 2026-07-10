# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for provider-neutral proof receipt recording helper (PROOF-RECEIPTS-1E)."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from typing import Optional

import pytest

from intergrax.integrations.contracts.document_store import (
    DocumentQueryResult,
    DocumentRecord,
    DocumentStore,
)
from intergrax.proofs.receipts.contracts import ProofReceipt, ProofReceiptResult
from intergrax.proofs.receipts.document_store import proof_receipt_to_document
from intergrax.proofs.receipts.recording import (
    ProofReceiptVerificationError,
    record_and_verify_proof_receipt,
    record_and_verify_proof_receipt_with_store,
    receipts_semantically_equal,
)
from intergrax.proofs.receipts.store import ProofReceiptStore

pytestmark = pytest.mark.unit


class _TrackingDocumentStore:
    """DocumentStore double that records lifecycle and supports controlled mismatches."""

    def __init__(self) -> None:
        self._documents: dict[tuple[str, str], DocumentRecord] = {}
        self.closed = False

    def get(self, partition_key: str, row_key: str) -> Optional[DocumentRecord]:
        return self._documents.get((partition_key, row_key))

    def put(self, document: DocumentRecord) -> None:
        self._documents[(document.partition_key, document.row_key)] = document

    def delete(self, partition_key: str, row_key: str) -> None:
        self._documents.pop((partition_key, row_key), None)

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: Optional[str] = None,
    ) -> DocumentQueryResult:
        matches: list[DocumentRecord] = []
        for (stored_partition, stored_row), document in sorted(self._documents.items()):
            if stored_partition != partition_key:
                continue
            if row_key_prefix is not None and not stored_row.startswith(row_key_prefix):
                continue
            matches.append(document)
        limited = matches[:limit]
        return DocumentQueryResult(documents=limited, total=len(matches))

    def close(self) -> None:
        self.closed = True


def _sample_receipt(*, result: ProofReceiptResult = ProofReceiptResult.PASS) -> ProofReceipt:
    return ProofReceipt(
        proof_id="local_workspace:platform_background_task:run-1",
        proof_kind="platform_background_task",
        application_id="local_workspace",
        result=result,
        recorded_at=datetime(2026, 7, 10, 10, 0, 0, tzinfo=UTC),
        run_id="run-1",
        correlation_id="corr-1",
        task_id="task-1",
        provider_evidence={"message_bus_provider": "kafka"},
        domain_evidence={"task_name": "lkw.background_ingest.v1"},
        guardrails={"mock_queue": False},
        metadata={"proof_runner": "run-lkw-background-task-proof.py"},
    )


def test_record_and_verify_persists_reads_back_and_queries() -> None:
    document_store = _TrackingDocumentStore()
    receipt = _sample_receipt()

    verified = record_and_verify_proof_receipt(receipt, document_store)  # type: ignore[arg-type]

    assert verified == receipt
    assert document_store.closed is True
    assert document_store.get(
        "proof_receipts/local_workspace",
        "proof/platform_background_task/run-1",
    ) is not None


def test_record_and_verify_raises_on_missing_read_back() -> None:
    document_store = _TrackingDocumentStore()

    class _MissingReadStore(_TrackingDocumentStore):
        def get(self, partition_key: str, row_key: str) -> Optional[DocumentRecord]:
            return None

    with pytest.raises(ProofReceiptVerificationError, match="read_back_missing"):
        record_and_verify_proof_receipt(_sample_receipt(), _MissingReadStore())  # type: ignore[arg-type]


def test_record_and_verify_raises_on_mismatched_read_back() -> None:
    document_store = _TrackingDocumentStore()
    receipt = _sample_receipt()

    class _MismatchReadStore(_TrackingDocumentStore):
        def get(self, partition_key: str, row_key: str) -> Optional[DocumentRecord]:
            mismatched = receipt.model_copy(update={"task_id": "different-task"})
            return proof_receipt_to_document(mismatched)

    with pytest.raises(ProofReceiptVerificationError, match="read_back_mismatch"):
        record_and_verify_proof_receipt(receipt, _MismatchReadStore())  # type: ignore[arg-type]


def test_record_and_verify_raises_when_query_missing_receipt() -> None:
    receipt = _sample_receipt()

    class _EmptyQueryStore(_TrackingDocumentStore):
        def query(
            self,
            partition_key: str,
            *,
            limit: int = 100,
            row_key_prefix: Optional[str] = None,
        ) -> DocumentQueryResult:
            return DocumentQueryResult(documents=[], total=0)

    with pytest.raises(ProofReceiptVerificationError, match="query_missing"):
        record_and_verify_proof_receipt(receipt, _EmptyQueryStore())  # type: ignore[arg-type]


def test_record_and_verify_preserves_fail_result() -> None:
    document_store = _TrackingDocumentStore()
    receipt = _sample_receipt(result=ProofReceiptResult.FAIL)

    verified = record_and_verify_proof_receipt(receipt, document_store)  # type: ignore[arg-type]

    assert verified.result == ProofReceiptResult.FAIL


def test_record_and_verify_with_store_leaves_lifecycle_to_caller() -> None:
    document_store = _TrackingDocumentStore()
    store = ProofReceiptStore(document_store)  # type: ignore[arg-type]
    receipt = _sample_receipt()

    verified = record_and_verify_proof_receipt_with_store(receipt, store)

    assert verified == receipt
    assert document_store.closed is False
    store.close()
    assert document_store.closed is True


def test_record_and_verify_can_skip_close_when_caller_owns_store() -> None:
    document_store = _TrackingDocumentStore()
    receipt = _sample_receipt()

    verified = record_and_verify_proof_receipt(
        receipt,
        document_store,  # type: ignore[arg-type]
        owns_document_store=False,
    )

    assert verified == receipt
    assert document_store.closed is False


def test_receipts_semantically_equal_compares_full_model() -> None:
    left = _sample_receipt()
    right = left.model_copy()
    assert receipts_semantically_equal(left, right) is True
    assert receipts_semantically_equal(left, left.model_copy(update={"task_id": "other"})) is False


def test_recording_module_has_no_vendor_or_lkw_imports() -> None:
    import intergrax.proofs.receipts.recording as recording_module

    source = inspect.getsource(recording_module)
    lowered = source.lower()
    assert "pymongo" not in lowered
    assert "kafka" not in lowered
    assert "local_workspace" not in lowered
