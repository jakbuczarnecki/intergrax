# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sanctioned worker-execution certified owner-loss issuer and staging (TOOL-LIFE-X L3-R7)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from intergrax.autonomous_work.repository import WorkerInstanceRepository
from intergrax.contracts.autonomous_work.ids import WorkerInstanceId
from intergrax.contracts.autonomous_work.lifecycle import WorkerLifecycleState
from intergrax.contracts.certified_owner_loss_evidence import (
    CANONICAL_WORKER_EXECUTION_OWNER_LOSS_ISSUER_ID,
    CertifiedOwnerLossDirectIssuerInvocationError,
    CertifiedOwnerLossEvidenceRejectedError,
    CompositionCertifiedOwnerLossIssuerBinding,
    CertifiedOwnerLossEvidenceV1,
    CertifiedOwnerLossIssuerNotConfiguredError,
    CertifiedOwnerLossReason,
    ExternalOperationOwnershipEpochV1,
    StagedCertifiedOwnerLossEvidenceV1,
    compute_certified_owner_loss_evidence_digest,
)
from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationNotFoundError,
    ExternalOperationPhysicalState,
    ExternalOperationState,
)
from intergrax.runtime.external_operations.certified_owner_loss_composition import (
    CertifiedOwnerLossComposition,
    _replace_certified_owner_loss_composition,
    require_certified_owner_loss_composition,
)
from intergrax.runtime.external_operations.external_operation_state_store import (
    ExternalOperationStateStore,
)

_WORKER_STILL_ALIVE_LIFECYCLE = frozenset(
    {
        WorkerLifecycleState.ACTIVE,
        WorkerLifecycleState.IDLE,
        WorkerLifecycleState.WORKING,
        WorkerLifecycleState.WAITING_EXTERNAL,
        WorkerLifecycleState.WAITING_FOR_HUMAN,
    },
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def worker_execution_liveness_proof_ref(
    *,
    worker_instance_id: str,
    bound_worker_ownership_epoch: int,
    observed_worker_revision: int,
    observed_lifecycle: WorkerLifecycleState,
) -> str:
    return (
        f"wexec-loss:v1:{worker_instance_id}:{bound_worker_ownership_epoch}"
        f"->{observed_worker_revision}:{observed_lifecycle.value}"
    )


def assert_worker_execution_ownership_lost(
    worker_repository: WorkerInstanceRepository,
    *,
    worker_instance_id: str,
    bound_worker_ownership_epoch: int,
) -> str:
    """Return opaque liveness proof when durable worker authority no longer matches binding."""
    worker = worker_repository.get(worker_instance_id=WorkerInstanceId(worker_instance_id))
    if worker is None:
        return worker_execution_liveness_proof_ref(
            worker_instance_id=worker_instance_id,
            bound_worker_ownership_epoch=bound_worker_ownership_epoch,
            observed_worker_revision=0,
            observed_lifecycle=WorkerLifecycleState.STOPPED,
        )
    observed_revision = worker.revision.value
    if (
        observed_revision <= bound_worker_ownership_epoch
        and worker.lifecycle_state in _WORKER_STILL_ALIVE_LIFECYCLE
    ):
        raise CertifiedOwnerLossEvidenceRejectedError(
            "worker still holds execution ownership for bound epoch",
        )
    return worker_execution_liveness_proof_ref(
        worker_instance_id=worker_instance_id,
        bound_worker_ownership_epoch=bound_worker_ownership_epoch,
        observed_worker_revision=observed_revision,
        observed_lifecycle=worker.lifecycle_state,
    )


class WorkerExecutionCertifiedOwnerLossEvidenceIssuer:
    """Exactly-one issuer binding; direct port calls are rejected."""

    def issue_owner_loss_evidence(
        self,
        *,
        epoch: ExternalOperationOwnershipEpochV1,
        loss_reason: CertifiedOwnerLossReason,
        liveness_proof_ref: str | None,
    ) -> CertifiedOwnerLossEvidenceV1:
        raise CertifiedOwnerLossDirectIssuerInvocationError(
            "use WorkerExecutionCertifiedOwnerLossCertifier",
        )

    def mint_sanctioned_evidence(
        self,
        *,
        epoch: ExternalOperationOwnershipEpochV1,
        loss_reason: CertifiedOwnerLossReason,
        liveness_proof_ref: str | None,
        monotonic_fence: int,
    ) -> CertifiedOwnerLossEvidenceV1:
        composition = require_certified_owner_loss_composition()
        if composition.issuer is not self:
            raise CertifiedOwnerLossIssuerNotConfiguredError(
                "issuer not registered in composition",
            )
        if CANONICAL_WORKER_EXECUTION_OWNER_LOSS_ISSUER_ID not in (
            composition.issuer_binding.allowed_issuer_ids
        ):
            raise CertifiedOwnerLossIssuerNotConfiguredError(
                "canonical worker-execution issuer not allowlisted",
            )
        if (
            loss_reason is CertifiedOwnerLossReason.LEASE_EXPIRED_WITH_LIVENESS_PROOF
            and not liveness_proof_ref
        ):
            raise CertifiedOwnerLossEvidenceRejectedError(
                "liveness_proof_ref required for lease-expired certification",
            )
        evidence = CertifiedOwnerLossEvidenceV1(
            evidence_id=f"col-{uuid.uuid4().hex}",
            epoch=epoch,
            issuer_id=CANONICAL_WORKER_EXECUTION_OWNER_LOSS_ISSUER_ID,
            issued_at=_utc_now(),
            monotonic_fence=monotonic_fence,
            loss_reason=loss_reason,
            liveness_proof_ref=liveness_proof_ref,
            evidence_digest="pending",
        )
        digest = compute_certified_owner_loss_evidence_digest(evidence)
        return evidence.model_copy(update={"evidence_digest": digest})


@dataclass(frozen=True, slots=True)
class WorkerExecutionCertifiedOwnerLossCertifier:
    """Non-bypassable staging path from worker ownership authority to W4-C store."""

    store: ExternalOperationStateStore
    worker_repository: WorkerInstanceRepository
    issuer: WorkerExecutionCertifiedOwnerLossEvidenceIssuer

    def stage_certified_owner_loss(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        loss_reason: CertifiedOwnerLossReason,
        liveness_proof_ref: str | None = None,
    ) -> ExternalOperationState:
        composition = require_certified_owner_loss_composition()
        if composition.worker_instance_repository is not self.worker_repository:
            raise CertifiedOwnerLossIssuerNotConfiguredError(
                "worker repository not registered in composition",
            )
        current = self.store.load(tenant_id, operation_id)
        if current is None:
            raise ExternalOperationNotFoundError(operation_id)
        if current.physical_state is not ExternalOperationPhysicalState.RUNNING:
            raise CertifiedOwnerLossEvidenceRejectedError("staging requires physical RUNNING")
        if current.worker_instance_id is None or current.worker_ownership_epoch is None:
            raise CertifiedOwnerLossEvidenceRejectedError(
                "worker execution ownership not bound on durable record",
            )
        proof = assert_worker_execution_ownership_lost(
            self.worker_repository,
            worker_instance_id=current.worker_instance_id,
            bound_worker_ownership_epoch=current.worker_ownership_epoch,
        )
        if liveness_proof_ref is not None and liveness_proof_ref != proof:
            raise CertifiedOwnerLossEvidenceRejectedError("liveness proof mismatch")
        effective_proof = liveness_proof_ref or proof
        next_fence = (current.ownership_epoch_fence or 0) + 1
        epoch = ExternalOperationOwnershipEpochV1(
            operation_id=current.operation_id,
            tenant_id=current.tenant_id,
            owner_token=current.owner_token or "",
            store_revision=current.revision,
        )
        evidence = self.issuer.mint_sanctioned_evidence(
            epoch=epoch,
            loss_reason=loss_reason,
            liveness_proof_ref=effective_proof,
            monotonic_fence=next_fence,
        )
        staged = StagedCertifiedOwnerLossEvidenceV1(
            evidence=evidence,
            staged_at=_utc_now(),
            staged_by_issuer_id=CANONICAL_WORKER_EXECUTION_OWNER_LOSS_ISSUER_ID,
        )
        return self.store.stage_certified_owner_loss_evidence(
            tenant_id,
            operation_id,
            expected_revision=current.revision,
            staged=staged,
        )

    def stage_and_apply_certified_owner_loss(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        loss_reason: CertifiedOwnerLossReason,
        liveness_proof_ref: str | None = None,
    ) -> ExternalOperationState:
        staged_state = self.stage_certified_owner_loss(
            tenant_id=tenant_id,
            operation_id=operation_id,
            loss_reason=loss_reason,
            liveness_proof_ref=liveness_proof_ref,
        )
        slot = staged_state.staged_certified_owner_loss
        if slot is None:
            raise CertifiedOwnerLossEvidenceRejectedError("staging did not persist evidence")
        return self.store.apply_certified_owner_loss(
            tenant_id,
            operation_id,
            expected_revision=staged_state.revision,
            evidence_id=slot.evidence.evidence_id,
        )


def configure_worker_execution_certified_owner_loss(
    *,
    worker_instance_repository: WorkerInstanceRepository,
    issuer: WorkerExecutionCertifiedOwnerLossEvidenceIssuer | None = None,
) -> None:
    """Register canonical worker-execution issuer binding for one deployment topology."""
    resolved_issuer = issuer or WorkerExecutionCertifiedOwnerLossEvidenceIssuer()
    _replace_certified_owner_loss_composition(
        CertifiedOwnerLossComposition(
            issuer_binding=CompositionCertifiedOwnerLossIssuerBinding(
                allowed_issuer_ids=frozenset(
                    {CANONICAL_WORKER_EXECUTION_OWNER_LOSS_ISSUER_ID},
                ),
            ),
            issuer=resolved_issuer,
            worker_instance_repository=worker_instance_repository,
        ),
    )


def build_worker_execution_certified_owner_loss_certifier(
    *,
    store: ExternalOperationStateStore,
    worker_instance_repository: WorkerInstanceRepository | None = None,
) -> WorkerExecutionCertifiedOwnerLossCertifier:
    composition = require_certified_owner_loss_composition()
    if not composition.issuer_binding.allowed_issuer_ids:
        raise CertifiedOwnerLossIssuerNotConfiguredError(
            "certified owner-loss issuer binding is not configured",
        )
    if composition.issuer is None or not isinstance(
        composition.issuer,
        WorkerExecutionCertifiedOwnerLossEvidenceIssuer,
    ):
        raise CertifiedOwnerLossIssuerNotConfiguredError("sanctioned issuer not configured")
    repo = worker_instance_repository or composition.worker_instance_repository
    if repo is None:
        raise CertifiedOwnerLossIssuerNotConfiguredError(
            "worker instance repository not configured",
        )
    return WorkerExecutionCertifiedOwnerLossCertifier(
        store=store,
        worker_repository=repo,
        issuer=composition.issuer,
    )


def bind_worker_execution_ownership(
    store: ExternalOperationStateStore,
    *,
    tenant_id: str,
    operation_id: str,
    expected_revision: int,
    worker_instance_id: str,
    worker_ownership_epoch: int,
) -> ExternalOperationState:
    """Bind durable worker execution identity to a RUNNING external operation."""
    return store.bind_worker_execution_ownership(
        tenant_id,
        operation_id,
        expected_revision=expected_revision,
        worker_instance_id=worker_instance_id,
        worker_ownership_epoch=worker_ownership_epoch,
    )


__all__ = [
    "WorkerExecutionCertifiedOwnerLossCertifier",
    "WorkerExecutionCertifiedOwnerLossEvidenceIssuer",
    "assert_worker_execution_ownership_lost",
    "bind_worker_execution_ownership",
    "build_worker_execution_certified_owner_loss_certifier",
    "configure_worker_execution_certified_owner_loss",
    "worker_execution_liveness_proof_ref",
]
