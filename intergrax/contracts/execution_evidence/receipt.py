# © Artur Czarnecki. All rights reserved.

"""Portable attested ProofReceipt (execution evidence — not DocumentStore LKW receipt)."""

from __future__ import annotations

import json
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.execution_evidence.attestation import HostAttestation
from intergrax.contracts.execution_evidence.boundary_event import (
    ExecutionBoundaryEvent,
    ExecutionBoundaryEventV2,
    SCHEMA_GOVERNED_EXECUTION_BOUNDARY_EVENT_V1,
    SCHEMA_GOVERNED_EXECUTION_BOUNDARY_EVENT_V2,
)
from intergrax.contracts.runtime_policy_bundle import ImmutableRuntimePolicyBundle

SCHEMA_EXECUTION_EVIDENCE_PROOF_RECEIPT_V1: Final = "execution_evidence.proof_receipt.v1"
SCHEMA_EXECUTION_EVIDENCE_PROOF_RECEIPT_V2: Final = "execution_evidence.proof_receipt.v2"
_NON_EMPTY = Field(min_length=1)


class ProofReceipt(BaseModel):
    """One portable attested export binding event + host attestation.

    Distinct from ``intergrax.proofs.receipts.ProofReceipt``
    (``intergrax.proof_receipt.v1`` DocumentStore persistence).

    Does not authorize execution. Immutable after signing.

    ``policy_bundle_artifact`` (PC-2 Model B) embeds the immutable pack body so
    offline verifiers can recompute digest without a network resolver.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["execution_evidence.proof_receipt.v1"] = (
        SCHEMA_EXECUTION_EVIDENCE_PROOF_RECEIPT_V1
    )
    receipt_id: str = _NON_EMPTY
    execution_boundary_event: ExecutionBoundaryEvent
    host_attestation: HostAttestation
    policy_bundle_artifact: ImmutableRuntimePolicyBundle | None = None

    @field_validator("receipt_id")
    @classmethod
    def _strip_receipt_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("receipt_id must be non-empty")
        return normalized


class ProofReceiptV2(BaseModel):
    """Portable attested export binding v2 boundary event + host attestation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["execution_evidence.proof_receipt.v2"] = (
        SCHEMA_EXECUTION_EVIDENCE_PROOF_RECEIPT_V2
    )
    receipt_id: str = _NON_EMPTY
    execution_boundary_event: ExecutionBoundaryEventV2
    host_attestation: HostAttestation
    policy_bundle_artifact: ImmutableRuntimePolicyBundle | None = None

    @field_validator("receipt_id")
    @classmethod
    def _strip_receipt_id_v2(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("receipt_id must be non-empty")
        return normalized


ExecutionEvidenceProofReceipt: TypeAlias = ProofReceipt | ProofReceiptV2


def parse_execution_evidence_proof_receipt_json(
    raw: str,
) -> ExecutionEvidenceProofReceipt:
    """Explicit schema dispatch for persisted execution-evidence receipts."""
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("proof_receipt_must_be_object")
    schema_id = data.get("schema_id")
    if schema_id == SCHEMA_EXECUTION_EVIDENCE_PROOF_RECEIPT_V1:
        return ProofReceipt.model_validate(data)
    if schema_id == SCHEMA_EXECUTION_EVIDENCE_PROOF_RECEIPT_V2:
        return ProofReceiptV2.model_validate(data)
    raise ValueError("unsupported_proof_receipt_schema")


def boundary_schema_for_proof_receipt(
    receipt: ExecutionEvidenceProofReceipt,
) -> str:
    if isinstance(receipt, ProofReceiptV2):
        return SCHEMA_GOVERNED_EXECUTION_BOUNDARY_EVENT_V2
    return SCHEMA_GOVERNED_EXECUTION_BOUNDARY_EVENT_V1
