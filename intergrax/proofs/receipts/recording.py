# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral proof receipt persistence with read-back verification (PROOF-RECEIPTS-1E)."""

from __future__ import annotations

from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.proofs.receipts.contracts import ProofReceipt
from intergrax.proofs.receipts.store import ProofReceiptStore


class ProofReceiptVerificationError(RuntimeError):
    """Raised when proof receipt write/read/query verification fails."""


def receipts_semantically_equal(left: ProofReceipt, right: ProofReceipt) -> bool:
    """Return True when two receipts represent the same persisted proof outcome."""
    return left == right


def record_and_verify_proof_receipt(
    receipt: ProofReceipt,
    document_store: DocumentStore,
    *,
    owns_document_store: bool = True,
) -> ProofReceipt:
    """
    Persist a proof receipt and verify read-back plus query presence.

    Lifecycle: when ``owns_document_store`` is True (default), this helper closes
    the underlying ``DocumentStore`` exactly once via ``ProofReceiptStore.close()``.
    When False, the caller retains lifecycle ownership and must close the store.
    """
    store = ProofReceiptStore(document_store)
    try:
        store.put(receipt)
        loaded = store.get(receipt.application_id, receipt.proof_kind, receipt.run_id)
        if loaded is None:
            raise ProofReceiptVerificationError("proof_receipt_read_back_missing")
        if not receipts_semantically_equal(receipt, loaded):
            raise ProofReceiptVerificationError("proof_receipt_read_back_mismatch")
        queried = store.query(receipt.application_id, proof_kind=receipt.proof_kind)
        if not any(item.run_id == receipt.run_id for item in queried):
            raise ProofReceiptVerificationError("proof_receipt_query_missing")
        return loaded
    finally:
        if owns_document_store:
            store.close()


def record_and_verify_proof_receipt_with_store(
    receipt: ProofReceipt,
    store: ProofReceiptStore,
) -> ProofReceipt:
    """
    Persist and verify using a caller-owned ``ProofReceiptStore``.

    The caller is responsible for closing the underlying document store.
    """
    store.put(receipt)
    loaded = store.get(receipt.application_id, receipt.proof_kind, receipt.run_id)
    if loaded is None:
        raise ProofReceiptVerificationError("proof_receipt_read_back_missing")
    if not receipts_semantically_equal(receipt, loaded):
        raise ProofReceiptVerificationError("proof_receipt_read_back_mismatch")
    queried = store.query(receipt.application_id, proof_kind=receipt.proof_kind)
    if not any(item.run_id == receipt.run_id for item in queried):
        raise ProofReceiptVerificationError("proof_receipt_query_missing")
    return loaded
