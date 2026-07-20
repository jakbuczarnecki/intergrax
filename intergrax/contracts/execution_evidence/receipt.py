# © Artur Czarnecki. All rights reserved.

"""Portable attested ProofReceipt (execution evidence — not DocumentStore LKW receipt)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.execution_evidence.attestation import HostAttestation
from intergrax.contracts.execution_evidence.boundary_event import ExecutionBoundaryEvent

SCHEMA_EXECUTION_EVIDENCE_PROOF_RECEIPT_V1: Final = "execution_evidence.proof_receipt.v1"
_NON_EMPTY = Field(min_length=1)


class ProofReceipt(BaseModel):
    """One portable attested export binding event + host attestation.

    Distinct from ``intergrax.proofs.receipts.ProofReceipt``
    (``intergrax.proof_receipt.v1`` DocumentStore persistence).

    Does not authorize execution. Immutable after signing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["execution_evidence.proof_receipt.v1"] = (
        SCHEMA_EXECUTION_EVIDENCE_PROOF_RECEIPT_V1
    )
    receipt_id: str = _NON_EMPTY
    execution_boundary_event: ExecutionBoundaryEvent
    host_attestation: HostAttestation

    @field_validator("receipt_id")
    @classmethod
    def _strip_receipt_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("receipt_id must be non-empty")
        return normalized
