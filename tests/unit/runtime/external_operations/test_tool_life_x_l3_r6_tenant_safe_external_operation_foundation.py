# © Artur Czarnecki. All rights reserved.

"""TOOL-LIFE-X L3-R6 — tenant-scoped store + certified-loss contract negatives."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.contracts.certified_owner_loss_evidence import (
    CertifiedOwnerLossAuthorityBypassError,
    CertifiedOwnerLossEvidenceRejectedError,
    CertifiedOwnerLossEvidenceV1,
    CertifiedOwnerLossIssuerNotConfiguredError,
    CertifiedOwnerLossReason,
    CompositionCertifiedOwnerLossIssuerBinding,
    ExternalOperationOwnershipEpochV1,
    StagedCertifiedOwnerLossEvidenceV1,
)
from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationIntentState,
    ExternalOperationPhysicalState,
)
from intergrax.runtime.external_operations.external_operation_ownership import (
    ExternalOperationCertifiedOwnerLossRequiredError,
    ProcessLocalExternalOperationOwner,
    mark_physical_running,
    mark_physical_terminal,
    recover_running_operation_after_certified_owner_loss,
    request_operation_cancellation,
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
_AUTHORIZED_ISSUER = "l3r6-test-issuer-binding"


def _bind_test_issuer(monkeypatch: pytest.MonkeyPatch) -> None:
    binding = CompositionCertifiedOwnerLossIssuerBinding(
        allowed_issuer_ids=frozenset({_AUTHORIZED_ISSUER}),
    )
    for target in (
        "intergrax.runtime.external_operations.certified_owner_loss_composition."
        "current_certified_owner_loss_issuer_binding",
        "intergrax.runtime.external_operations.external_operation_state_store."
        "current_certified_owner_loss_issuer_binding",
    ):
        monkeypatch.setattr(target, lambda: binding)


def _evidence_for_running(
    running: object,
    owner: ProcessLocalExternalOperationOwner,
    *,
    evidence_id: str = "ev-authorized",
    fence: int = 1,
) -> CertifiedOwnerLossEvidenceV1:
    return CertifiedOwnerLossEvidenceV1(
        evidence_id=evidence_id,
        epoch=ExternalOperationOwnershipEpochV1(
            operation_id=_OP,
            tenant_id=_TENANT_A,
            owner_token=owner.token,
            store_revision=running.revision,
        ),
        issuer_id=_AUTHORIZED_ISSUER,
        issued_at=datetime.now(timezone.utc),
        monotonic_fence=fence,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
        evidence_digest="aabbcc",
    )


def _stage(
    store: InMemoryExternalOperationStateStore,
    running: object,
    owner: ProcessLocalExternalOperationOwner,
    *,
    evidence_id: str = "ev-authorized",
) -> object:
    evidence = _evidence_for_running(running, owner, evidence_id=evidence_id)
    staged = StagedCertifiedOwnerLossEvidenceV1(
        evidence=evidence,
        staged_at=datetime.now(timezone.utc),
        staged_by_issuer_id=_AUTHORIZED_ISSUER,
    )
    return store.stage_certified_owner_loss_evidence(
        _TENANT_A,
        _OP,
        expected_revision=running.revision,
        staged=staged,
    )


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


def test_l3r6_stage_apply_revision_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_test_issuer(monkeypatch)
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    _record, running = _running_record(store, tenant_id=_TENANT_A, owner=owner)
    assert running.revision == _record.revision + 1
    staged_state = _stage(store, running, owner)
    assert staged_state.revision == running.revision + 1
    assert staged_state.staged_certified_owner_loss is not None
    assert (
        staged_state.staged_certified_owner_loss.evidence.epoch.store_revision
        == running.revision
    )
    applied = store.apply_certified_owner_loss(
        _TENANT_A,
        _OP,
        expected_revision=staged_state.revision,
        evidence_id="ev-authorized",
    )
    assert applied.physical_state is ExternalOperationPhysicalState.UNKNOWN
    assert applied.revision == staged_state.revision + 1
    assert applied.staged_certified_owner_loss is None
    assert "ev-authorized" in applied.consumed_certified_loss_evidence_ids


def test_l3r6_apply_idempotent_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_test_issuer(monkeypatch)
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    _record, running = _running_record(store, tenant_id=_TENANT_A, owner=owner)
    staged_state = _stage(store, running, owner)
    applied = store.apply_certified_owner_loss(
        _TENANT_A,
        _OP,
        expected_revision=staged_state.revision,
        evidence_id="ev-authorized",
    )
    again = store.apply_certified_owner_loss(
        _TENANT_A,
        _OP,
        expected_revision=applied.revision,
        evidence_id="ev-authorized",
    )
    assert again.revision == applied.revision
    assert again.physical_state is ExternalOperationPhysicalState.UNKNOWN


def test_l3r6_stale_pre_stage_revision_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_test_issuer(monkeypatch)
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    record, running = _running_record(store, tenant_id=_TENANT_A, owner=owner)
    evidence = _evidence_for_running(running, owner)
    evidence = evidence.model_copy(
        update={
            "epoch": evidence.epoch.model_copy(
                update={"store_revision": record.revision},
            ),
        },
    )
    staged = StagedCertifiedOwnerLossEvidenceV1(
        evidence=evidence,
        staged_at=datetime.now(timezone.utc),
        staged_by_issuer_id=_AUTHORIZED_ISSUER,
    )
    with pytest.raises(CertifiedOwnerLossEvidenceRejectedError):
        store.stage_certified_owner_loss_evidence(
            _TENANT_A,
            _OP,
            expected_revision=running.revision,
            staged=staged,
        )
    loaded = store.load(_TENANT_A, _OP)
    assert loaded is not None
    assert loaded.revision == running.revision
    assert loaded.staged_certified_owner_loss is None


def test_l3r6_apply_with_pre_stage_revision_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_test_issuer(monkeypatch)
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    _record, running = _running_record(store, tenant_id=_TENANT_A, owner=owner)
    staged_state = _stage(store, running, owner)
    with pytest.raises(CertifiedOwnerLossEvidenceRejectedError):
        store.apply_certified_owner_loss(
            _TENANT_A,
            _OP,
            expected_revision=running.revision,
            evidence_id="ev-authorized",
        )
    loaded = store.load(_TENANT_A, _OP)
    assert loaded is not None
    assert loaded.revision == staged_state.revision
    assert loaded.physical_state is ExternalOperationPhysicalState.RUNNING


def test_l3r6_concurrent_staging_one_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    _bind_test_issuer(monkeypatch)
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    _record, running = _running_record(store, tenant_id=_TENANT_A, owner=owner)
    errors: list[BaseException] = []

    def attempt(evidence_id: str) -> None:
        try:
            _stage(store, running, owner, evidence_id=evidence_id)
        except BaseException as exc:  # noqa: BLE001 — adversarial concurrency
            errors.append(exc)

    t1 = threading.Thread(target=attempt, args=("ev-a",))
    t2 = threading.Thread(target=attempt, args=("ev-b",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    loaded = store.load(_TENANT_A, _OP)
    assert loaded is not None
    assert loaded.revision == running.revision + 1
    assert loaded.staged_certified_owner_loss is not None
    assert len(errors) >= 1


def test_l3r6_restaging_after_revision_advance_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_test_issuer(monkeypatch)
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    _record, running = _running_record(store, tenant_id=_TENANT_A, owner=owner)
    staged_state = _stage(store, running, owner)
    with pytest.raises(CertifiedOwnerLossEvidenceRejectedError):
        _stage(store, staged_state, owner, evidence_id="ev-other")
    loaded = store.load(_TENANT_A, _OP)
    assert loaded is not None
    assert loaded.staged_certified_owner_loss is not None
    assert loaded.staged_certified_owner_loss.evidence.evidence_id == "ev-authorized"


def test_l3r6_terminal_race_before_apply_preserves_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_test_issuer(monkeypatch)
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    _record, running = _running_record(store, tenant_id=_TENANT_A, owner=owner)
    staged_state = _stage(store, running, owner)
    terminal = mark_physical_terminal(
        store,
        tenant_id=_TENANT_A,
        operation_id=_OP,
        expected_revision=staged_state.revision,
        physical_state=ExternalOperationPhysicalState.SUCCEEDED,
    )
    with pytest.raises(CertifiedOwnerLossEvidenceRejectedError):
        store.apply_certified_owner_loss(
            _TENANT_A,
            _OP,
            expected_revision=terminal.revision,
            evidence_id="ev-authorized",
        )
    loaded = store.load(_TENANT_A, _OP)
    assert loaded is not None
    assert loaded.physical_state is ExternalOperationPhysicalState.SUCCEEDED


def test_l3r6_cancellation_does_not_apply_certified_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_test_issuer(monkeypatch)
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    _record, running = _running_record(store, tenant_id=_TENANT_A, owner=owner)
    staged_state = _stage(store, running, owner)
    cancelled = request_operation_cancellation(
        store,
        tenant_id=_TENANT_A,
        operation_id=_OP,
    )
    assert cancelled.intent_state is ExternalOperationIntentState.CANCELLATION_REQUESTED
    with pytest.raises(CertifiedOwnerLossEvidenceRejectedError):
        store.apply_certified_owner_loss(
            _TENANT_A,
            _OP,
            expected_revision=cancelled.revision,
            evidence_id="ev-authorized",
        )
    loaded = store.load(_TENANT_A, _OP)
    assert loaded is not None
    assert loaded.revision == cancelled.revision
    assert loaded.physical_state is ExternalOperationPhysicalState.RUNNING


def test_l3r6_forged_staged_by_issuer_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_test_issuer(monkeypatch)
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    _record, running = _running_record(store, tenant_id=_TENANT_A, owner=owner)
    evidence = _evidence_for_running(running, owner)
    staged = StagedCertifiedOwnerLossEvidenceV1(
        evidence=evidence,
        staged_at=datetime.now(timezone.utc),
        staged_by_issuer_id="forged-stager",
    )
    with pytest.raises(CertifiedOwnerLossIssuerNotConfiguredError):
        store.stage_certified_owner_loss_evidence(
            _TENANT_A,
            _OP,
            expected_revision=running.revision,
            staged=staged,
        )


def test_l3r6_unauthorized_cas_cannot_stage_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_test_issuer(monkeypatch)
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    _record, running = _running_record(store, tenant_id=_TENANT_A, owner=owner)
    evidence = _evidence_for_running(running, owner)
    staged = StagedCertifiedOwnerLossEvidenceV1(
        evidence=evidence,
        staged_at=datetime.now(timezone.utc),
        staged_by_issuer_id=_AUTHORIZED_ISSUER,
    )
    with pytest.raises(CertifiedOwnerLossAuthorityBypassError):
        store.compare_and_set(
            _TENANT_A,
            _OP,
            expected_revision=running.revision,
            update=ExternalOperationStateUpdate(
                staged_certified_owner_loss=staged,
            ),
        )


def test_l3r6_unauthorized_cas_cannot_consume_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_test_issuer(monkeypatch)
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    _record, running = _running_record(store, tenant_id=_TENANT_A, owner=owner)
    with pytest.raises(CertifiedOwnerLossAuthorityBypassError):
        store.compare_and_set(
            _TENANT_A,
            _OP,
            expected_revision=running.revision,
            update=ExternalOperationStateUpdate(
                consumed_certified_loss_evidence_ids=frozenset({"forged"}),
            ),
        )


def test_l3r6_unauthorized_cas_cannot_set_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    _record, running = _running_record(store, tenant_id=_TENANT_A, owner=owner)
    with pytest.raises(CertifiedOwnerLossAuthorityBypassError):
        store.compare_and_set(
            _TENANT_A,
            _OP,
            expected_revision=running.revision,
            update=ExternalOperationStateUpdate(ownership_epoch_fence=99),
        )


def test_l3r6_unauthorized_cas_cannot_certified_unknown_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    _record, running = _running_record(store, tenant_id=_TENANT_A, owner=owner)
    with pytest.raises(CertifiedOwnerLossAuthorityBypassError):
        store.compare_and_set(
            _TENANT_A,
            _OP,
            expected_revision=running.revision,
            update=ExternalOperationStateUpdate(
                physical_state=ExternalOperationPhysicalState.UNKNOWN,
                clear_owner_token=True,
                ownership_epoch_fence=1,
            ),
        )
    loaded = store.load(_TENANT_A, _OP)
    assert loaded is not None
    assert loaded.physical_state is ExternalOperationPhysicalState.RUNNING


def test_l3r6_provider_uncertainty_unknown_via_lifecycle_allowed() -> None:
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    _record, running = _running_record(store, tenant_id=_TENANT_A, owner=owner)
    unknown = mark_physical_terminal(
        store,
        tenant_id=_TENANT_A,
        operation_id=_OP,
        expected_revision=running.revision,
        physical_state=ExternalOperationPhysicalState.UNKNOWN,
    )
    assert unknown.physical_state is ExternalOperationPhysicalState.UNKNOWN


def test_l3r6_concurrent_apply_one_consumes_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    _bind_test_issuer(monkeypatch)
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    _record, running = _running_record(store, tenant_id=_TENANT_A, owner=owner)
    staged_state = _stage(store, running, owner)
    errors: list[BaseException] = []
    results: list[object] = []

    def attempt() -> None:
        try:
            results.append(
                store.apply_certified_owner_loss(
                    _TENANT_A,
                    _OP,
                    expected_revision=staged_state.revision,
                    evidence_id="ev-authorized",
                ),
            )
        except BaseException as exc:  # noqa: BLE001 — adversarial concurrency
            errors.append(exc)

    t1 = threading.Thread(target=attempt)
    t2 = threading.Thread(target=attempt)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    loaded = store.load(_TENANT_A, _OP)
    assert loaded is not None
    assert loaded.physical_state is ExternalOperationPhysicalState.UNKNOWN
    assert "ev-authorized" in loaded.consumed_certified_loss_evidence_ids
    assert len(results) + len(errors) == 2


def test_l3r6_owner_change_after_stage_rejects_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_test_issuer(monkeypatch)
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    _record, running = _running_record(store, tenant_id=_TENANT_A, owner=owner)
    staged_state = _stage(store, running, owner)
    new_owner = ProcessLocalExternalOperationOwner.mint()
    store.compare_and_set(
        _TENANT_A,
        _OP,
        expected_revision=staged_state.revision,
        update=ExternalOperationStateUpdate(owner_token=new_owner.token),
    )
    loaded = store.load(_TENANT_A, _OP)
    assert loaded is not None
    with pytest.raises(CertifiedOwnerLossEvidenceRejectedError):
        store.apply_certified_owner_loss(
            _TENANT_A,
            _OP,
            expected_revision=loaded.revision,
            evidence_id="ev-authorized",
        )


def test_l3r6_blind_retry_after_unknown_denied() -> None:
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    _record, running = _running_record(store, tenant_id=_TENANT_A, owner=owner)
    unknown = mark_physical_terminal(
        store,
        tenant_id=_TENANT_A,
        operation_id=_OP,
        expected_revision=running.revision,
        physical_state=ExternalOperationPhysicalState.UNKNOWN,
    )
    result = recover_running_operation_after_certified_owner_loss(
        store,
        tenant_id=_TENANT_A,
        operation_id=_OP,
        expected_revision=unknown.revision,
    )
    assert result.physical_state is ExternalOperationPhysicalState.UNKNOWN
    assert result.revision == unknown.revision
