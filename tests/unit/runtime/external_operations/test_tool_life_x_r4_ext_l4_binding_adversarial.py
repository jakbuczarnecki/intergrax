# © Artur Czarnecki. All rights reserved.

"""EXT-L4 B1 — durable trusted-contract binding adversarial qualification."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.autonomous_work.in_memory_repository import InMemoryWorkerInstanceRepository
from intergrax.contracts.certified_owner_loss_evidence import CertifiedOwnerLossReason
from intergrax.contracts.external_operation_redrive_effect_binding import (
    ExternalOperationRedriveEffectBindingError,
    ExternalOperationRedriveEffectBindingMismatchError,
)
from intergrax.contracts.external_operation_redrive_trusted_effect_contracts import (
    TOOL_LIFE_HTTP_BLIND_POST_QUAL_EFFECT,
    TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT,
    TOOL_LIFE_PLATFORM_DURABLE_PHYSICAL_LIFECYCLE_EFFECT,
)
from intergrax.contracts.external_operation_redrive_authority import (
    ExternalOperationRedriveCompetingAuthorityError,
    ExternalOperationRedriveEffectClassRejectedError,
)
from intergrax.runtime.external_operations.recovery_external_operation_gate import (
    ExternalOperationReconciliationAdmissionError,
)
from intergrax.runtime.external_operations.redrive_external_operation_authority import (
    bind_authoritative_redrive_effect_to_operation,
    transfer_redrive_authority_after_reconciliation,
)
from intergrax.runtime.external_operations.external_operation_state_store import (
    StaleExternalOperationStateError,
)
from intergrax.runtime.external_operations.sqlite_tool_life_durable_stores import (
    SqliteToolLifeDurableStoresBundle,
)
from intergrax.runtime.external_operations.worker_execution_certified_owner_loss import (
    build_worker_execution_certified_owner_loss_certifier,
)
from intergrax.runtime.external_operations.external_operation_ownership import (
    recover_running_operation_after_certified_owner_loss,
)
from tests.unit.autonomous_work.repository_contracts import worker_instance
from tests.unit.runtime.external_operations.test_tool_life_x_l3_r8_sqlite_durable_revocation_shared_store import (
    _BOUND_EPOCH,
    _OP,
    _TENANT,
    _revoke_bound,
    _running_with_worker,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_OTHER_TENANT = "tenant-other"
_BIND_CLASS_C = {
    "bind_redrive_contract_id": TOOL_LIFE_HTTP_BLIND_POST_QUAL_EFFECT.contract_id,
    "bind_provider_id": "http_rest",
    "bind_tool_id": "http_blind_post",
}
_BIND_CLASS_B = {
    "bind_redrive_contract_id": TOOL_LIFE_PLATFORM_DURABLE_PHYSICAL_LIFECYCLE_EFFECT.contract_id,
    "bind_provider_id": "platform",
    "bind_tool_id": "external_operation_physical_lifecycle",
}


@pytest.fixture
def durable_bundle(tmp_path: Path) -> SqliteToolLifeDurableStoresBundle:
    return SqliteToolLifeDurableStoresBundle(db_path=tmp_path / "binding-adversarial.db")


def _unknown_after_loss(bundle: SqliteToolLifeDurableStoresBundle) -> object:
    repo = InMemoryWorkerInstanceRepository()
    _running_with_worker(bundle, repo, **_BIND_CLASS_C)
    proof = _revoke_bound(bundle)
    certifier = build_worker_execution_certified_owner_loss_certifier(
        store=bundle.external_operation_state,
    )
    staged = certifier.stage_certified_owner_loss(
        tenant_id=_TENANT,
        operation_id=_OP,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
        liveness_proof_ref=proof,
    )
    return recover_running_operation_after_certified_owner_loss(
        bundle.external_operation_state,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=staged.revision,
        evidence_id=staged.staged_certified_owner_loss.evidence.evidence_id,
    )


def test_substitute_class_b_contract_for_class_c_bound_operation_rejected(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    unknown = _unknown_after_loss(durable_bundle)
    recovery_worker = worker_instance()
    InMemoryWorkerInstanceRepository().create(recovery_worker)
    with pytest.raises(ExternalOperationRedriveEffectBindingMismatchError):
        transfer_redrive_authority_after_reconciliation(
            durable_bundle.external_operation_state,
            tenant_id=_TENANT,
            operation_id=_OP,
            expected_revision=unknown.revision,
            trusted_effect_contract=TOOL_LIFE_PLATFORM_DURABLE_PHYSICAL_LIFECYCLE_EFFECT,
            worker_instance_id=str(recovery_worker.worker_instance_id),
            worker_ownership_epoch=_BOUND_EPOCH + 1,
            provider_id="platform",
            tool_id="external_operation_physical_lifecycle",
        )


def test_mismatch_contract_operation_key_rejected(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    unknown = _unknown_after_loss(durable_bundle)
    forged = TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT.model_copy(
        update={"contract_id": TOOL_LIFE_HTTP_BLIND_POST_QUAL_EFFECT.contract_id},
    )
    recovery_worker = worker_instance()
    with pytest.raises(
        (ExternalOperationRedriveEffectClassRejectedError, ExternalOperationRedriveEffectBindingMismatchError),
    ):
        transfer_redrive_authority_after_reconciliation(
            durable_bundle.external_operation_state,
            tenant_id=_TENANT,
            operation_id=_OP,
            expected_revision=unknown.revision,
            trusted_effect_contract=forged,
            worker_instance_id=str(recovery_worker.worker_instance_id),
            worker_ownership_epoch=_BOUND_EPOCH + 1,
            provider_id="http_rest",
            tool_id="http_blind_post",
        )


def test_cross_tenant_redrive_substitution_rejected(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    unknown = _unknown_after_loss(durable_bundle)
    recovery_worker = worker_instance()
    with pytest.raises(ExternalOperationRedriveEffectBindingError):
        transfer_redrive_authority_after_reconciliation(
            durable_bundle.external_operation_state,
            tenant_id=_OTHER_TENANT,
            operation_id=_OP,
            expected_revision=unknown.revision,
            trusted_effect_contract=TOOL_LIFE_HTTP_BLIND_POST_QUAL_EFFECT,
            worker_instance_id=str(recovery_worker.worker_instance_id),
            worker_ownership_epoch=_BOUND_EPOCH + 1,
            provider_id="http_rest",
            tool_id="http_blind_post",
        )


def test_stale_store_revision_transfer_rejected(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    _running_with_worker(durable_bundle, repo, **_BIND_CLASS_B)
    proof = _revoke_bound(durable_bundle)
    certifier = build_worker_execution_certified_owner_loss_certifier(
        store=durable_bundle.external_operation_state,
    )
    staged = certifier.stage_certified_owner_loss(
        tenant_id=_TENANT,
        operation_id=_OP,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
        liveness_proof_ref=proof,
    )
    unknown = recover_running_operation_after_certified_owner_loss(
        durable_bundle.external_operation_state,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=staged.revision,
        evidence_id=staged.staged_certified_owner_loss.evidence.evidence_id,
    )
    recovery_worker = worker_instance()
    repo.create(recovery_worker)
    with pytest.raises(StaleExternalOperationStateError):
        transfer_redrive_authority_after_reconciliation(
            durable_bundle.external_operation_state,
            tenant_id=_TENANT,
            operation_id=_OP,
            expected_revision=unknown.revision - 1,
            trusted_effect_contract=TOOL_LIFE_PLATFORM_DURABLE_PHYSICAL_LIFECYCLE_EFFECT,
            worker_instance_id=str(recovery_worker.worker_instance_id),
            worker_ownership_epoch=_BOUND_EPOCH + 1,
            provider_id="platform",
            tool_id="external_operation_physical_lifecycle",
        )


def test_competing_redrive_authority_still_rejected(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    _running_with_worker(durable_bundle, repo, **_BIND_CLASS_B)
    proof = _revoke_bound(durable_bundle)
    certifier = build_worker_execution_certified_owner_loss_certifier(
        store=durable_bundle.external_operation_state,
    )
    staged = certifier.stage_certified_owner_loss(
        tenant_id=_TENANT,
        operation_id=_OP,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
        liveness_proof_ref=proof,
    )
    unknown = recover_running_operation_after_certified_owner_loss(
        durable_bundle.external_operation_state,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=staged.revision,
        evidence_id=staged.staged_certified_owner_loss.evidence.evidence_id,
    )
    worker_a = worker_instance()
    worker_b = worker_instance()
    repo.create(worker_a)
    repo.create(worker_b)
    transfer_redrive_authority_after_reconciliation(
        durable_bundle.external_operation_state,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=unknown.revision,
        trusted_effect_contract=TOOL_LIFE_PLATFORM_DURABLE_PHYSICAL_LIFECYCLE_EFFECT,
        worker_instance_id=str(worker_a.worker_instance_id),
        worker_ownership_epoch=_BOUND_EPOCH + 1,
        provider_id="platform",
        tool_id="external_operation_physical_lifecycle",
    )
    loaded = durable_bundle.external_operation_state.load(_TENANT, _OP)
    assert loaded is not None
    with pytest.raises(ExternalOperationRedriveCompetingAuthorityError):
        durable_bundle.external_operation_state.transfer_redrive_worker_execution_authority(
            tenant_id=_TENANT,
            operation_id=_OP,
            expected_revision=loaded.revision,
            worker_instance_id=str(worker_b.worker_instance_id),
            worker_ownership_epoch=_BOUND_EPOCH + 2,
            redrive_admission_id="redrive-admit:competing-worker",
        )


def test_forged_reconciliation_without_unknown_admission_rejected(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    store = durable_bundle.external_operation_state
    record = store.create_if_absent(_TENANT, "op-no-unknown")
    bind_authoritative_redrive_effect_to_operation(
        store,
        tenant_id=_TENANT,
        operation_id="op-no-unknown",
        expected_revision=record.revision,
        trusted_effect_contract_id=TOOL_LIFE_PLATFORM_DURABLE_PHYSICAL_LIFECYCLE_EFFECT.contract_id,
        provider_id="platform",
        tool_id="external_operation_physical_lifecycle",
    )
    with pytest.raises(ExternalOperationReconciliationAdmissionError):
        transfer_redrive_authority_after_reconciliation(
            store,
            tenant_id=_TENANT,
            operation_id="op-no-unknown",
            expected_revision=record.revision,
            trusted_effect_contract=TOOL_LIFE_PLATFORM_DURABLE_PHYSICAL_LIFECYCLE_EFFECT,
            worker_instance_id="recovery-worker",
            worker_ownership_epoch=2,
            provider_id="platform",
            tool_id="external_operation_physical_lifecycle",
        )
