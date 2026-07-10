# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for ProofReceipt contracts and DocumentStore-backed store (PROOF-RECEIPTS-1A)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Optional

import pytest

from intergrax.integrations.contracts.document_store import (
    DocumentQueryResult,
    DocumentRecord,
    DocumentStore,
)
from intergrax.proofs.receipts.contracts import (
    PROOF_RECEIPT_PARTITION_PREFIX,
    PROOF_RECEIPT_SCHEMA_VERSION,
    ProofReceipt,
    ProofReceiptResult,
)
from intergrax.proofs.receipts.document_store import (
    proof_receipt_from_document,
    proof_receipt_partition_key,
    proof_receipt_row_key,
    proof_receipt_to_document,
)
from intergrax.proofs.receipts.store import ProofReceiptStore

pytestmark = pytest.mark.unit


class _InMemoryDocumentStoreForUnitTests:
    """Minimal DocumentStore double for unit-testing ProofReceiptStore only."""

    def __init__(self) -> None:
        self._documents: dict[tuple[str, str], DocumentRecord] = {}

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
        return None


def _lkw_4e_receipt() -> ProofReceipt:
    recorded_at = datetime(2026, 7, 9, 10, 42, 59, tzinfo=UTC)
    return ProofReceipt(
        proof_id="proof_lkw_4e_kafka_20260709104259",
        proof_kind="platform_background_task",
        application_id="local_workspace_application",
        result=ProofReceiptResult.PASS,
        recorded_at=recorded_at,
        run_id="lkw-bg-ingest-run-20260709104259",
        correlation_id="corr-lkw-4e-20260709104259",
        task_id="task-kafka-20260709104259",
        provider_evidence={
            "message_bus_provider": "kafka",
            "document_store_provider": "mongodb",
        },
        domain_evidence={
            "collection_id": "local_workspace",
            "marker": "lkw-background-task-proof-marker",
            "search_results": 1,
            "evidence_marker_found": True,
        },
        guardrails={
            "mock_queue": False,
            "inmemory_bypass": False,
            "direct_handler_call": False,
            "direct_indexer_call": False,
        },
        metadata={"proof_track": "LKW.4E"},
    )


def test_proof_receipt_represents_lkw_4e_evidence_shape() -> None:
    receipt = _lkw_4e_receipt()

    assert receipt.schema_version == PROOF_RECEIPT_SCHEMA_VERSION
    assert receipt.proof_kind == "platform_background_task"
    assert receipt.application_id == "local_workspace_application"
    assert receipt.result == ProofReceiptResult.PASS
    assert receipt.run_id == "lkw-bg-ingest-run-20260709104259"
    assert receipt.correlation_id == "corr-lkw-4e-20260709104259"
    assert receipt.task_id == "task-kafka-20260709104259"
    assert receipt.provider_evidence["message_bus_provider"] == "kafka"
    assert receipt.provider_evidence["document_store_provider"] == "mongodb"
    assert receipt.domain_evidence["collection_id"] == "local_workspace"
    assert receipt.domain_evidence["marker"] == "lkw-background-task-proof-marker"
    assert receipt.domain_evidence["search_results"] == 1
    assert receipt.domain_evidence["evidence_marker_found"] is True
    assert receipt.guardrails["mock_queue"] is False
    assert receipt.guardrails["inmemory_bypass"] is False
    assert receipt.guardrails["direct_handler_call"] is False
    assert receipt.guardrails["direct_indexer_call"] is False


def test_proof_receipt_to_document_maps_to_document_record() -> None:
    receipt = _lkw_4e_receipt()
    document = proof_receipt_to_document(receipt)

    assert document.partition_key == (
        f"{PROOF_RECEIPT_PARTITION_PREFIX}/local_workspace_application"
    )
    assert document.row_key == (
        "proof/platform_background_task/lkw-bg-ingest-run-20260709104259"
    )
    assert document.data == receipt.model_dump(mode="json")
    assert document.ttl_seconds is None


def test_proof_receipt_from_document_round_trips() -> None:
    receipt = _lkw_4e_receipt()
    document = proof_receipt_to_document(receipt)

    restored = proof_receipt_from_document(document)

    assert restored == receipt


def test_proof_receipt_store_put_get_query_and_close() -> None:
    document_store = _InMemoryDocumentStoreForUnitTests()
    store = ProofReceiptStore(document_store)  # type: ignore[arg-type]
    receipt = _lkw_4e_receipt()
    other_receipt = receipt.model_copy(
        update={
            "proof_id": "proof_other_kind",
            "proof_kind": "platform_observability",
            "run_id": "run-obs-1",
        }
    )

    store.put(receipt)
    store.put(other_receipt)

    loaded = store.get(
        "local_workspace_application",
        "platform_background_task",
        "lkw-bg-ingest-run-20260709104259",
    )
    assert loaded == receipt

    filtered = store.query(
        "local_workspace_application",
        proof_kind="platform_background_task",
    )
    assert filtered == [receipt]

    all_receipts = store.query("local_workspace_application")
    assert len(all_receipts) == 2

    store.close()


def test_proof_receipt_partition_key_is_stable() -> None:
    assert proof_receipt_partition_key("local_workspace_application") == (
        f"{PROOF_RECEIPT_PARTITION_PREFIX}/local_workspace_application"
    )


def test_proof_receipt_row_key_is_stable() -> None:
    receipt = _lkw_4e_receipt()
    assert proof_receipt_row_key(receipt) == (
        "proof/platform_background_task/lkw-bg-ingest-run-20260709104259"
    )
