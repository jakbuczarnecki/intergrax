# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""DocumentStore-backed proof receipt persistence engine (PROOF-RECEIPTS-1A)."""

from __future__ import annotations

from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.proofs.receipts.contracts import ProofReceipt
from intergrax.proofs.receipts.document_store import (
    proof_receipt_from_document,
    proof_receipt_lookup_row_key,
    proof_receipt_partition_key,
    proof_receipt_row_key_prefix,
    proof_receipt_to_document,
)


class ProofReceiptStore:
    """Provider-neutral proof receipt store delegating to DocumentStore."""

    def __init__(self, document_store: DocumentStore) -> None:
        self._document_store = document_store

    def put(self, receipt: ProofReceipt) -> None:
        self._document_store.put(proof_receipt_to_document(receipt))

    def get(
        self,
        application_id: str,
        proof_kind: str,
        run_id: str,
    ) -> ProofReceipt | None:
        partition_key = proof_receipt_partition_key(application_id)
        row_key = proof_receipt_lookup_row_key(proof_kind, run_id)
        document = self._document_store.get(partition_key, row_key)
        if document is None:
            return None
        return proof_receipt_from_document(document)

    def query(
        self,
        application_id: str,
        *,
        proof_kind: str | None = None,
        limit: int = 100,
    ) -> list[ProofReceipt]:
        partition_key = proof_receipt_partition_key(application_id)
        row_key_prefix = (
            proof_receipt_row_key_prefix(proof_kind) if proof_kind is not None else None
        )
        result = self._document_store.query(
            partition_key,
            limit=limit,
            row_key_prefix=row_key_prefix,
        )
        receipts: list[ProofReceipt] = []
        for document in result.documents:
            receipts.append(proof_receipt_from_document(document))
        return receipts

    def close(self) -> None:
        self._document_store.close()
