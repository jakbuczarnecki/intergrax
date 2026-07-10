# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Structured proof receipt contracts and DocumentStore-backed persistence."""

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

__all__ = [
    "PROOF_RECEIPT_PARTITION_PREFIX",
    "PROOF_RECEIPT_SCHEMA_VERSION",
    "ProofReceipt",
    "ProofReceiptResult",
    "ProofReceiptStore",
    "proof_receipt_from_document",
    "proof_receipt_partition_key",
    "proof_receipt_row_key",
    "proof_receipt_to_document",
]
