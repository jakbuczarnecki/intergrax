# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed certified owner-loss contracts (TOOL-LIFE-X L3-R6 — foundation only).

Contracts and store protocol types only. No sanctioned issuer implementation;
positive RUNNING→UNKNOWN recovery remains fail-closed until composition binding.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class CertifiedOwnerLossReason(StrEnum):
    WORKER_LOST = "WORKER_LOST"
    LEASE_EXPIRED_WITH_LIVENESS_PROOF = "LEASE_EXPIRED_WITH_LIVENESS_PROOF"
    OPERATOR_DECLARED = "OPERATOR_DECLARED"


class ExternalOperationOwnershipEpochV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    owner_token: str = Field(min_length=1)
    store_revision: int = Field(ge=1)


class CertifiedOwnerLossEvidenceV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    epoch: ExternalOperationOwnershipEpochV1
    issuer_id: str = Field(min_length=1)
    issued_at: datetime
    monotonic_fence: int = Field(ge=0)
    loss_reason: CertifiedOwnerLossReason
    liveness_proof_ref: str | None = None
    evidence_digest: str = Field(min_length=1)


class StagedCertifiedOwnerLossEvidenceV1(BaseModel):
    """Durable staged slot — written only by sanctioned issuer composition path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence: CertifiedOwnerLossEvidenceV1
    staged_at: datetime
    staged_by_issuer_id: str = Field(min_length=1)


class CompositionCertifiedOwnerLossIssuerBinding(BaseModel):
    """Composition-scoped allowlist; empty binding keeps recovery fail-closed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed_issuer_ids: frozenset[str] = frozenset()


class CertifiedOwnerLossEvidenceRejectedError(RuntimeError):
    """Evidence failed store-native validation or consumption rules."""


class CertifiedOwnerLossIssuerNotConfiguredError(RuntimeError):
    """No sanctioned issuer binding — staging and apply remain unavailable."""


class CertifiedOwnerLossAuthorityBypassError(RuntimeError):
    """Ordinary lifecycle CAS attempted to mutate certification authority fields."""


@runtime_checkable
class CertifiedOwnerLossEvidenceIssuerPort(Protocol):
    def issue_owner_loss_evidence(
        self,
        *,
        epoch: ExternalOperationOwnershipEpochV1,
        loss_reason: CertifiedOwnerLossReason,
        liveness_proof_ref: str | None,
    ) -> CertifiedOwnerLossEvidenceV1: ...
