# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed certified owner-loss contracts (TOOL-LIFE-X L3-R6/L3-R7).

Store-native stage/apply authority types; sanctioned issuer wired via composition.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

CANONICAL_WORKER_EXECUTION_OWNER_LOSS_ISSUER_ID = (
    "tool-life-x.worker-execution-ownership.v1"
)


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


class CertifiedOwnerLossDirectIssuerInvocationError(RuntimeError):
    """Issuer port invoked outside sanctioned worker-execution composition path."""


def compute_certified_owner_loss_evidence_digest(
    evidence: CertifiedOwnerLossEvidenceV1,
) -> str:
    """Integrity fingerprint for staged bytes — not issuer authenticity alone."""
    payload = {
        "evidence_id": evidence.evidence_id,
        "epoch": {
            "operation_id": evidence.epoch.operation_id,
            "tenant_id": evidence.epoch.tenant_id,
            "owner_token": evidence.epoch.owner_token,
            "store_revision": evidence.epoch.store_revision,
        },
        "issuer_id": evidence.issuer_id,
        "issued_at": evidence.issued_at.isoformat(),
        "monotonic_fence": evidence.monotonic_fence,
        "loss_reason": evidence.loss_reason.value,
        "liveness_proof_ref": evidence.liveness_proof_ref,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@runtime_checkable
class CertifiedOwnerLossEvidenceIssuerPort(Protocol):
    def issue_owner_loss_evidence(
        self,
        *,
        epoch: ExternalOperationOwnershipEpochV1,
        loss_reason: CertifiedOwnerLossReason,
        liveness_proof_ref: str | None,
    ) -> CertifiedOwnerLossEvidenceV1: ...
