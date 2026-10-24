# © Artur Czarnecki. All rights reserved.

"""TOOL-LIFE-X L3-R6 — tenant-scoped store + certified-loss contract negatives."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.contracts.certified_owner_loss_evidence import (
    CertifiedOwnerLossEvidenceRejectedError,
    CertifiedOwnerLossEvidenceV1,
    CertifiedOwnerLossIssuerNotConfiguredError,
    CertifiedOwnerLossReason,
    ExternalOperationOwnershipEpochV1,
    StagedCertifiedOwnerLossEvidenceV1,
)
from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationPhysicalState,
)
from intergrax.runtime.external_operations.external_operation_ownership import (
    ExternalOperationCertifiedOwnerLossRequiredError,
    ProcessLocalExternalOperationOwner,
    mark_physical_running,
    recover_running_operation_after_certified_owner_loss,
)
from intergrax.contracts.external_operation_cancellation import ExternalOperationNotFoundError
from intergrax.runtime.external_operations.external_operation_state_store import (
    ExternalOperationLegacyTenantBindingError,
    ExternalOperationStateUpdate,
    InMemoryExternalOperationStateStore,
    StaleExternalOperationStateError,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT_A = "tenant_l3r6_a"
_TENANT_B = "tenant_l3r6_b"
_OP = "shared-operation-id"


def _running_record(
    store: InMemoryExternalOperationStateStore,
    *,
    tenant_id: str,
    owner: ProcessLocalExternalOperationOwner,
) -> tuple[object, object]:
    record = store.create_if_absent(tenant_id, _OP, owner_token=owner.token)
    running = mark_physical_running(
        store,
        tenant_id=tenant_id,
        operation_id=_OP,
        expected_revision=record.revision,
        owner=owner,
    )
    return record, running


def test_l3r6_tenant_a_cannot_load_tenant_b_operation() -> None:
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    store.create_if_absent(_TENANT_A, _OP, owner_token=owner.token)
    assert store.load(_TENANT_B, _OP) is None


def test_l3r6_tenant_a_cannot_cas_tenant_b_operation() -> None:
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    record = store.create_if_absent(_TENANT_B, _OP, owner_token=owner.token)
    with pytest.raises(ExternalOperationNotFoundError):
        store.compare_and_set(
            _TENANT_A,
            _OP,
            expected_revision=record.revision,
            update=ExternalOperationStateUpdate(
                physical_state=ExternalOperationPhysicalState.RUNNING,
            ),
        )


def test_l3r6_same_operation_id_isolated_per_tenant() -> None:
    store = InMemoryExternalOperationStateStore()
    owner_a = ProcessLocalExternalOperationOwner.mint()
    owner_b = ProcessLocalExternalOperationOwner.mint()
    store.create_if_absent(_TENANT_A, _OP, owner_token=owner_a.token)
    store.create_if_absent(_TENANT_B, _OP, owner_token=owner_b.token)
    a = store.load(_TENANT_A, _OP)
    b = store.load(_TENANT_B, _OP)
    assert a is not None and b is not None
    assert a.owner_token != b.owner_token
    assert a.tenant_id == _TENANT_A
    assert b.tenant_id == _TENANT_B


def test_l3r6_list_running_is_tenant_scoped() -> None:
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    _running_record(store, tenant_id=_TENANT_A, owner=owner)
    assert len(store.list_running(_TENANT_A)) == 1
    assert store.list_running(_TENANT_B) == ()


def test_l3r6_legacy_record_without_tenant_rejected() -> None:
    from intergrax.contracts.external_operation_cancellation import (
        ExternalOperationIntentState,
        ExternalOperationState,
    )

    store = InMemoryExternalOperationStateStore()
    now = datetime.now(timezone.utc)
    legacy = ExternalOperationState(
        operation_id=_OP,
        tenant_id="",
        intent_state=ExternalOperationIntentState.ACTIVE,
        physical_state=ExternalOperationPhysicalState.NOT_STARTED,
        created_at=now,
        updated_at=now,
        revision=1,
    )
    store._records[(_TENANT_A, _OP)] = legacy  # noqa: SLF001 — adversarial inject
    with pytest.raises(ExternalOperationLegacyTenantBindingError):
        store.load(_TENANT_A, _OP)


def test_l3r6_forged_certification_apply_rejected() -> None:
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    record, running = _running_record(store, tenant_id=_TENANT_A, owner=owner)
    with pytest.raises(CertifiedOwnerLossIssuerNotConfiguredError):
        store.apply_certified_owner_loss(
            _TENANT_A,
            _OP,
            expected_revision=running.revision,
            evidence_id="forged-evidence",
        )
    loaded = store.load(_TENANT_A, _OP)
    assert loaded is not None
    assert loaded.physical_state is ExternalOperationPhysicalState.RUNNING
    assert loaded.revision == running.revision


def test_l3r6_no_issuer_binding_recovery_api_denied() -> None:
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    _record, running = _running_record(store, tenant_id=_TENANT_A, owner=owner)
    with pytest.raises(ExternalOperationCertifiedOwnerLossRequiredError):
        recover_running_operation_after_certified_owner_loss(
            store,
            tenant_id=_TENANT_A,
            operation_id=_OP,
            expected_revision=running.revision,
        )


def test_l3r6_stage_without_issuer_binding_denied() -> None:
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    _record, running = _running_record(store, tenant_id=_TENANT_A, owner=owner)
    epoch = ExternalOperationOwnershipEpochV1(
        operation_id=_OP,
        tenant_id=_TENANT_A,
        owner_token=owner.token,
        store_revision=running.revision,
    )
    evidence = CertifiedOwnerLossEvidenceV1(
        evidence_id="ev-1",
        epoch=epoch,
        issuer_id="forged-issuer",
        issued_at=datetime.now(timezone.utc),
        monotonic_fence=1,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
        evidence_digest="deadbeef",
    )
    staged = StagedCertifiedOwnerLossEvidenceV1(
        evidence=evidence,
        staged_at=datetime.now(timezone.utc),
        staged_by_issuer_id="forged-issuer",
    )
    with pytest.raises(CertifiedOwnerLossIssuerNotConfiguredError):
        store.stage_certified_owner_loss_evidence(
            _TENANT_A,
            _OP,
            expected_revision=running.revision,
            staged=staged,
        )


def test_l3r6_cross_tenant_certification_cannot_apply() -> None:
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    _record, running = _running_record(store, tenant_id=_TENANT_A, owner=owner)
    with pytest.raises(CertifiedOwnerLossIssuerNotConfiguredError):
        store.apply_certified_owner_loss(
            _TENANT_B,
            _OP,
            expected_revision=running.revision,
            evidence_id="ev-cross",
        )


def test_l3r6_failed_cas_leaves_running_unchanged() -> None:
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    _record, running = _running_record(store, tenant_id=_TENANT_A, owner=owner)
    with pytest.raises(CertifiedOwnerLossIssuerNotConfiguredError):
        store.apply_certified_owner_loss(
            _TENANT_A,
            _OP,
            expected_revision=running.revision + 99,
            evidence_id="ev-stale",
        )
    loaded = store.load(_TENANT_A, _OP)
    assert loaded is not None
    assert loaded.physical_state is ExternalOperationPhysicalState.RUNNING
    assert loaded.revision == running.revision


def test_l3r6_stale_cas_does_not_partially_mutate() -> None:
    from intergrax.runtime.external_operations.external_operation_state_store import (
        ExternalOperationStateUpdate,
    )

    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    record = store.create_if_absent(_TENANT_A, _OP, owner_token=owner.token)
    with pytest.raises(StaleExternalOperationStateError):
        store.compare_and_set(
            _TENANT_A,
            _OP,
            expected_revision=record.revision + 5,
            update=ExternalOperationStateUpdate(
                physical_state=ExternalOperationPhysicalState.RUNNING,
            ),
        )
    loaded = store.load(_TENANT_A, _OP)
    assert loaded is not None
    assert loaded.revision == record.revision
