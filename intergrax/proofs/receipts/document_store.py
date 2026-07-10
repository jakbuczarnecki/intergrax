# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ProofReceipt ↔ DocumentRecord mapping (provider-neutral)."""

from __future__ import annotations

from typing import Any

from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.proofs.receipts.contracts import (
    PROOF_RECEIPT_PARTITION_PREFIX,
    PROOF_RECEIPT_SCHEMA_VERSION,
    ProofReceipt,
)

_PROOF_ROW_PREFIX = "proof"


def proof_receipt_partition_key(application_id: str) -> str:
    """Stable partition scoped by proof workload application."""
    normalized = application_id.strip()
    if not normalized:
        raise ValueError("application_id must not be blank")
    return f"{PROOF_RECEIPT_PARTITION_PREFIX}/{normalized}"


def proof_receipt_lookup_row_key(proof_kind: str, run_id: str) -> str:
    """Stable row key for a proof kind and run identifier."""
    normalized_kind = proof_kind.strip()
    normalized_run_id = run_id.strip()
    if not normalized_kind:
        raise ValueError("proof_kind must not be blank")
    if not normalized_run_id:
        raise ValueError("run_id must not be blank")
    return f"{_PROOF_ROW_PREFIX}/{normalized_kind}/{normalized_run_id}"


def proof_receipt_row_key(receipt: ProofReceipt) -> str:
    """Stable, queryable row key within the application partition."""
    return proof_receipt_lookup_row_key(receipt.proof_kind, receipt.run_id)


def proof_receipt_row_key_prefix(proof_kind: str) -> str:
    """Row-key prefix for filtering receipts by proof kind within a partition."""
    normalized = proof_kind.strip()
    if not normalized:
        raise ValueError("proof_kind must not be blank")
    return f"{_PROOF_ROW_PREFIX}/{normalized}/"


def proof_receipt_to_document(receipt: ProofReceipt) -> DocumentRecord:
    """Map a proof receipt to a provider-neutral document row."""
    return DocumentRecord(
        partition_key=proof_receipt_partition_key(receipt.application_id),
        row_key=proof_receipt_row_key(receipt),
        data=receipt.model_dump(mode="json"),
        ttl_seconds=None,
    )


def proof_receipt_from_document(document: DocumentRecord) -> ProofReceipt:
    """Rehydrate a proof receipt from a document row."""
    data: dict[str, Any] = dict(document.data)
    schema_version = data.get("schema_version")
    if schema_version != PROOF_RECEIPT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported proof receipt schema_version: {schema_version!r}"
        )
    return ProofReceipt.model_validate(data)
