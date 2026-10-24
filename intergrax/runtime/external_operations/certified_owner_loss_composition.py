# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Composition binding for certified owner-loss issuers (fail-closed default)."""

from __future__ import annotations

from dataclasses import dataclass
from intergrax.autonomous_work.repository import WorkerInstanceRepository
from intergrax.contracts.certified_owner_loss_evidence import (
    CANONICAL_WORKER_EXECUTION_OWNER_LOSS_ISSUER_ID,
    CertifiedOwnerLossEvidenceIssuerPort,
    CertifiedOwnerLossEvidenceRejectedError,
    CertifiedOwnerLossIssuerNotConfiguredError,
    CompositionCertifiedOwnerLossIssuerBinding,
)
from intergrax.contracts.external_operation_cancellation import ExternalOperationState
from intergrax.contracts.worker_execution_ownership_revocation import (
    WorkerExecutionOwnershipRevocationPort,
)

_DEFAULT_BINDING = CompositionCertifiedOwnerLossIssuerBinding()


@dataclass(frozen=True, slots=True)
class CertifiedOwnerLossComposition:
    issuer_binding: CompositionCertifiedOwnerLossIssuerBinding
    issuer: CertifiedOwnerLossEvidenceIssuerPort | None = None
    worker_instance_repository: WorkerInstanceRepository | None = None
    worker_execution_ownership_revocation: WorkerExecutionOwnershipRevocationPort | None = (
        None
    )


_COMPOSITION = CertifiedOwnerLossComposition(issuer_binding=_DEFAULT_BINDING)


def current_certified_owner_loss_issuer_binding() -> CompositionCertifiedOwnerLossIssuerBinding:
    """Return deployment issuer allowlist; empty until operator composition wires issuers."""
    return _COMPOSITION.issuer_binding


def require_certified_owner_loss_composition() -> CertifiedOwnerLossComposition:
    return _COMPOSITION


def verify_worker_execution_owner_loss_staging(
    *,
    current: ExternalOperationState,
    liveness_proof_ref: str | None,
) -> None:
    """Store-native guard: worker-execution evidence must cite authoritative revocation."""
    if liveness_proof_ref is None or not liveness_proof_ref.strip():
        raise CertifiedOwnerLossEvidenceRejectedError(
            "worker-execution certified loss requires revocation proof ref",
        )
    composition = require_certified_owner_loss_composition()
    port = composition.worker_execution_ownership_revocation
    if port is None:
        raise CertifiedOwnerLossIssuerNotConfiguredError(
            "worker execution ownership revocation port is not configured",
        )
    if current.worker_instance_id is None or current.worker_ownership_epoch is None:
        raise CertifiedOwnerLossEvidenceRejectedError(
            "worker execution ownership not bound on durable record",
        )
    expected = port.authoritative_revocation_proof_ref(
        tenant_id=current.tenant_id,
        operation_id=current.operation_id,
        worker_instance_id=current.worker_instance_id,
        ownership_epoch=current.worker_ownership_epoch,
    )
    if expected is None or expected != liveness_proof_ref:
        raise CertifiedOwnerLossEvidenceRejectedError(
            "liveness proof does not match authoritative revocation record",
        )


def _replace_certified_owner_loss_composition(
    composition: CertifiedOwnerLossComposition,
) -> None:
    global _COMPOSITION
    _COMPOSITION = composition


def reset_certified_owner_loss_composition_for_tests() -> None:
    """Test-only reset to fail-closed default."""
    _replace_certified_owner_loss_composition(
        CertifiedOwnerLossComposition(issuer_binding=_DEFAULT_BINDING),
    )


def is_worker_execution_issuer_id(issuer_id: str) -> bool:
    return issuer_id == CANONICAL_WORKER_EXECUTION_OWNER_LOSS_ISSUER_ID
