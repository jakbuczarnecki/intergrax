# © Artur Czarnecki. All rights reserved.

"""TOOL-LIFE-X L3-R8 — SQLite durable revocation + shared external-operation store."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.autonomous_work.in_memory_repository import InMemoryWorkerInstanceRepository
from intergrax.contracts.worker_execution_ownership_revocation import (
    WorkerExecutionDirectRevocationInvocationError,
    WorkerExecutionOwnershipRevocationAuthorizationRejectedError,
    WorkerExecutionOwnershipRevocationSupervisionNotConfiguredError,
    WorkerExecutionOwnershipRevocationUnauthorizedError,
)
from tests.unit.runtime.external_operations.tool_life_x_qualification_revocation_supervision import (
    QualificationWorkerExecutionRevocationSupervision,
)
from tests.unit.runtime.external_operations.tool_life_x_revocation_test_forgery import (
    forged_worker_execution_ownership_revocation_authorization,
)
from intergrax.runtime.external_operations.worker_execution_ownership_revocation_trust import (
    reset_worker_execution_ownership_revocation_trust_for_tests,
)
from intergrax.runtime.external_operations.certified_owner_loss_composition import (
    require_certified_owner_loss_composition,
)
from intergrax.contracts.certified_owner_loss_evidence import (
    CANONICAL_WORKER_EXECUTION_OWNER_LOSS_ISSUER_ID,
    CertifiedOwnerLossEvidenceRejectedError,
    CertifiedOwnerLossIssuerNotConfiguredError,
    CertifiedOwnerLossReason,
)
from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationIntentState,
    ExternalOperationNotFoundError,
    ExternalOperationPhysicalState,
)
from intergrax.contracts.persistence_topology import PersistenceTopology
from intergrax.runtime.external_operations.certified_owner_loss_composition import (
    reset_certified_owner_loss_composition_for_tests,
)
from intergrax.runtime.external_operations.external_operation_ownership import (
    ExternalOperationFencedCommitRejectedError,
    ProcessLocalExternalOperationOwner,
    mark_physical_running,
    mark_physical_terminal_if_owner_matches,
    recover_running_operation_after_certified_owner_loss,
)
from intergrax.runtime.external_operations.external_operation_state_store import (
    StaleExternalOperationStateError,
)
from intergrax.runtime.external_operations.sqlite_tool_life_durable_stores import (
    SqliteToolLifeDurableStoresBundle,
    SQLiteExternalOperationStateStore,
    SQLiteWorkerExecutionOwnershipRevocationStore,
)
from tests.unit.runtime.external_operations.sqlite_tool_life_l3_r8_cross_process_proof import (
    run_revocation_wins_before_terminal_race,
    run_terminal_wins_before_revocation_race,
)
from intergrax.runtime.external_operations.redrive_external_operation_authority import (
    bind_authoritative_redrive_effect_to_operation,
)
from intergrax.runtime.external_operations.worker_execution_certified_owner_loss import (
    bind_worker_execution_ownership,
    build_worker_execution_certified_owner_loss_certifier,
    configure_worker_execution_certified_owner_loss,
)
from intergrax.runtime.external_operations.worker_execution_ownership_revocation import (
    build_worker_execution_ownership_revocation_recorder,
)
from tests.unit.autonomous_work.repository_contracts import worker_instance
from tests.unit.runtime.external_operations.tool_life_x_test_commit_authority_loss import (
    issue_sqlite_commit_authority_loss_decision_for_bound_operation,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant_l3r8"
_OP = "op-l3r8"
_BOUND_EPOCH = 11


@pytest.fixture(autouse=True)
def _reset_composition() -> None:
    reset_certified_owner_loss_composition_for_tests()
    reset_worker_execution_ownership_revocation_trust_for_tests()
    yield
    reset_certified_owner_loss_composition_for_tests()
    reset_worker_execution_ownership_revocation_trust_for_tests()


@pytest.fixture
def durable_bundle(tmp_path: Path) -> SqliteToolLifeDurableStoresBundle:
    return SqliteToolLifeDurableStoresBundle(db_path=tmp_path / "tool_life_durable.db")


def _running_with_worker(
    bundle: SqliteToolLifeDurableStoresBundle,
    repo: InMemoryWorkerInstanceRepository,
    *,
    ownership_epoch: int = _BOUND_EPOCH,
    bind_redrive_contract_id: str | None = None,
    bind_provider_id: str | None = None,
    bind_tool_id: str | None = None,
    bind_physical_idempotency_key: str | None = None,
    bind_physical_effect_fingerprint: str | None = None,
) -> tuple[object, object, str, ProcessLocalExternalOperationOwner]:
    store = bundle.external_operation_state
    revocation = bundle.worker_execution_ownership_revocation
    worker = worker_instance()
    repo.create(worker)
    owner = ProcessLocalExternalOperationOwner.mint()
    record = store.create_if_absent(_TENANT, _OP, owner_token=owner.token)
    if bind_redrive_contract_id is not None:
        record = bind_authoritative_redrive_effect_to_operation(
            store,
            tenant_id=_TENANT,
            operation_id=_OP,
            expected_revision=record.revision,
            trusted_effect_contract_id=bind_redrive_contract_id,
            provider_id=bind_provider_id,
            tool_id=bind_tool_id,
            physical_idempotency_key=bind_physical_idempotency_key,
            physical_effect_fingerprint=bind_physical_effect_fingerprint,
        )
    running = mark_physical_running(
        store,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=record.revision,
        owner=owner,
    )
    bound = bind_worker_execution_ownership(
        store,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=running.revision,
        worker_instance_id=str(worker.worker_instance_id),
        worker_ownership_epoch=ownership_epoch,
    )
    configure_worker_execution_certified_owner_loss(
        worker_execution_ownership_revocation=revocation,
        worker_instance_repository=repo,
        authoritative_state_store=store,
        worker_execution_revocation_supervision=QualificationWorkerExecutionRevocationSupervision(),
    )
    return worker, bound, str(worker.worker_instance_id), owner


def _revoke_bound(
    bundle: SqliteToolLifeDurableStoresBundle,
    *,
    operation_id: str = _OP,
) -> str:
    issue_sqlite_commit_authority_loss_decision_for_bound_operation(
        bundle,
        tenant_id=_TENANT,
        operation_id=operation_id,
    )
    recorder = require_certified_owner_loss_composition().worker_execution_revocation_recorder
    assert recorder is not None
    return recorder.record_revocation_for_bound_operation(
        tenant_id=_TENANT,
        operation_id=operation_id,
    ).proof_ref


def test_l3r8_bundle_topology_durable_single_host(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    assert durable_bundle.persistence_topology is PersistenceTopology.DURABLE_SINGLE_HOST
    assert (
        durable_bundle.external_operation_state.persistence_topology
        is PersistenceTopology.DURABLE_SINGLE_HOST
    )
    assert (
        durable_bundle.worker_execution_ownership_revocation.persistence_topology
        is PersistenceTopology.DURABLE_SINGLE_HOST
    )


def test_l3r8_positive_revoke_stage_apply_survives_store_reopen(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
    tmp_path: Path,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    _worker, bound, worker_id, _owner = _running_with_worker(durable_bundle, repo)
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
    evidence_id = staged.staged_certified_owner_loss.evidence.evidence_id
    reopened = SqliteToolLifeDurableStoresBundle(db_path=durable_bundle.db_path)
    unknown = recover_running_operation_after_certified_owner_loss(
        reopened.external_operation_state,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=staged.revision,
        evidence_id=evidence_id,
    )
    assert unknown.physical_state is ExternalOperationPhysicalState.UNKNOWN
    assert unknown.intent_state is ExternalOperationIntentState.TERMINATING
    persisted = reopened.external_operation_state.load(_TENANT, _OP)
    assert persisted is not None
    assert persisted.revision == unknown.revision
    assert evidence_id in persisted.consumed_certified_loss_evidence_ids


def test_l3r8_idempotent_revocation_record(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    _worker, _bound, worker_id, _owner = _running_with_worker(durable_bundle, repo)
    port = durable_bundle.worker_execution_ownership_revocation
    first_proof = _revoke_bound(durable_bundle)
    second_proof = _revoke_bound(durable_bundle)
    assert second_proof == first_proof
    assert port.authoritative_revocation_proof_ref(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id=worker_id,
        ownership_epoch=_BOUND_EPOCH,
    ) == first_proof


def test_l3r8_staging_without_revocation_rejected(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    _running_with_worker(durable_bundle, repo)
    certifier = build_worker_execution_certified_owner_loss_certifier(
        store=durable_bundle.external_operation_state,
    )
    with pytest.raises(CertifiedOwnerLossEvidenceRejectedError):
        certifier.stage_certified_owner_loss(
            tenant_id=_TENANT,
            operation_id=_OP,
            loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
        )


def test_l3r8_wrong_epoch_revocation_denied(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    _worker, bound, worker_id, _owner = _running_with_worker(durable_bundle, repo)
    forged = forged_worker_execution_ownership_revocation_authorization(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id=worker_id,
        ownership_epoch=_BOUND_EPOCH + 1,
        bound_store_revision=bound.revision,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
    )
    with pytest.raises(WorkerExecutionDirectRevocationInvocationError):
        durable_bundle.worker_execution_ownership_revocation.revoke_worker_execution_ownership(
            tenant_id=_TENANT,
            operation_id=_OP,
            worker_instance_id=worker_id,
            ownership_epoch=_BOUND_EPOCH + 1,
            authorization=forged,
        )
    certifier = build_worker_execution_certified_owner_loss_certifier(
        store=durable_bundle.external_operation_state,
    )
    with pytest.raises(CertifiedOwnerLossEvidenceRejectedError):
        certifier.stage_certified_owner_loss(
            tenant_id=_TENANT,
            operation_id=_OP,
            loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
        )


def test_l3r8_cross_tenant_revocation_not_visible(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    _worker, _bound, worker_id, _owner = _running_with_worker(durable_bundle, repo)
    current = durable_bundle.external_operation_state.load(_TENANT, _OP)
    assert current is not None
    forged = forged_worker_execution_ownership_revocation_authorization(
        tenant_id="other-tenant",
        operation_id=_OP,
        worker_instance_id=worker_id,
        ownership_epoch=_BOUND_EPOCH,
        bound_store_revision=current.revision,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
    )
    with pytest.raises(WorkerExecutionDirectRevocationInvocationError):
        durable_bundle.worker_execution_ownership_revocation.revoke_worker_execution_ownership(
            tenant_id="other-tenant",
            operation_id=_OP,
            worker_instance_id=worker_id,
            ownership_epoch=_BOUND_EPOCH,
            authorization=forged,
        )
    proof = durable_bundle.worker_execution_ownership_revocation.authoritative_revocation_proof_ref(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id=worker_id,
        ownership_epoch=_BOUND_EPOCH,
    )
    assert proof is None


def test_l3r8_concurrent_cas_stale_revision(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    store = durable_bundle.external_operation_state
    record = store.create_if_absent(_TENANT, "op-cas")
    owner = ProcessLocalExternalOperationOwner.mint()
    running = mark_physical_running(
        store,
        tenant_id=_TENANT,
        operation_id="op-cas",
        expected_revision=record.revision,
        owner=owner,
    )
    with pytest.raises(StaleExternalOperationStateError):
        mark_physical_running(
            store,
            tenant_id=_TENANT,
            operation_id="op-cas",
            expected_revision=record.revision,
            owner=owner,
        )
    assert running.revision == record.revision + 1


def test_l3r8_separate_connection_revocation_visibility(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    """WAL-backed SQLite: independent connection observes committed revocation."""
    repo = InMemoryWorkerInstanceRepository()
    _worker, _bound, worker_id, _owner = _running_with_worker(durable_bundle, repo)
    proof = _revoke_bound(durable_bundle)
    peer = SQLiteWorkerExecutionOwnershipRevocationStore(
        db_path=durable_bundle.db_path,
    )
    peer_proof = peer.authoritative_revocation_proof_ref(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id=worker_id,
        ownership_epoch=_BOUND_EPOCH,
    )
    assert peer_proof == proof


def test_l3r8_separate_connection_external_operation_state(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    _running_with_worker(durable_bundle, repo)
    peer = SQLiteExternalOperationStateStore(db_path=durable_bundle.db_path)
    loaded = peer.load(_TENANT, _OP)
    assert loaded is not None
    assert loaded.physical_state is ExternalOperationPhysicalState.RUNNING


def test_l3r8_issuer_allowlist_required_for_stage(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    _worker, _bound, worker_id, _owner = _running_with_worker(durable_bundle, repo)
    _revoke_bound(durable_bundle)
    reset_certified_owner_loss_composition_for_tests()
    with pytest.raises(CertifiedOwnerLossIssuerNotConfiguredError):
        build_worker_execution_certified_owner_loss_certifier(
            store=durable_bundle.external_operation_state,
        ).stage_certified_owner_loss(
            tenant_id=_TENANT,
            operation_id=_OP,
            loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
        )


def test_l3r8_unauthorized_revocation_without_binding_rejected(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    durable_bundle.external_operation_state.create_if_absent(_TENANT, "op-unbound")
    forged = forged_worker_execution_ownership_revocation_authorization(
        tenant_id=_TENANT,
        operation_id="op-unbound",
        worker_instance_id="worker-x",
        ownership_epoch=1,
        bound_store_revision=1,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
    )
    with pytest.raises(WorkerExecutionDirectRevocationInvocationError):
        durable_bundle.worker_execution_ownership_revocation.revoke_worker_execution_ownership(
            tenant_id=_TENANT,
            operation_id="op-unbound",
            worker_instance_id="worker-x",
            ownership_epoch=1,
            authorization=forged,
        )


def test_l3r8_fenced_worker_terminal_after_revocation(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    _worker, bound, worker_id, owner = _running_with_worker(durable_bundle, repo)
    _revoke_bound(durable_bundle)
    with pytest.raises(ExternalOperationFencedCommitRejectedError):
        mark_physical_terminal_if_owner_matches(
            durable_bundle.external_operation_state,
            tenant_id=_TENANT,
            operation_id=_OP,
            expected_revision=bound.revision,
            physical_state=ExternalOperationPhysicalState.SUCCEEDED,
            owner=owner,
        )


def test_l3r8_concurrent_revocation_idempotent(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    import threading

    repo = InMemoryWorkerInstanceRepository()
    _worker, _bound, worker_id, _owner = _running_with_worker(durable_bundle, repo)
    port = durable_bundle.worker_execution_ownership_revocation
    issue_sqlite_commit_authority_loss_decision_for_bound_operation(
        durable_bundle,
        tenant_id=_TENANT,
        operation_id=_OP,
    )
    results: list[str] = []
    errors: list[BaseException] = []

    def _revoke() -> None:
        try:
            recorder = require_certified_owner_loss_composition().worker_execution_revocation_recorder
            assert recorder is not None
            record = recorder.record_revocation_for_bound_operation(
                tenant_id=_TENANT,
                operation_id=_OP,
            )
            results.append(record.proof_ref)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_revoke) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(results) == 4
    assert len(set(results)) == 1


def test_l3r8_canonical_issuer_id_unchanged() -> None:
    assert (
        CANONICAL_WORKER_EXECUTION_OWNER_LOSS_ISSUER_ID
        == "tool-life-x.worker-execution-ownership.v1"
    )


def test_l3r8_direct_port_revocation_without_sanction_rejected(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    _running_with_worker(durable_bundle, repo)
    forged = forged_worker_execution_ownership_revocation_authorization(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id="ignored",
        ownership_epoch=_BOUND_EPOCH,
        bound_store_revision=1,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
    )
    with pytest.raises(WorkerExecutionDirectRevocationInvocationError):
        durable_bundle.worker_execution_ownership_revocation.revoke_worker_execution_ownership(
            tenant_id=_TENANT,
            operation_id=_OP,
            worker_instance_id="ignored",
            ownership_epoch=_BOUND_EPOCH,
            authorization=forged,
        )


def test_l3r8_cross_process_revocation_wins_before_terminal_cas(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    worker, bound, worker_id, _owner = _running_with_worker(durable_bundle, repo)
    result = run_revocation_wins_before_terminal_race(
        db_path=durable_bundle.db_path,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=bound.revision,
    )
    assert len(result.outcomes) == 2
    assert all(code == 0 for code in result.exit_codes)
    assert "revoked" in result.outcomes
    assert "fenced" in result.outcomes
    loaded = durable_bundle.external_operation_state.load(_TENANT, _OP)
    assert loaded is not None
    assert loaded.physical_state is ExternalOperationPhysicalState.RUNNING
    proof = durable_bundle.worker_execution_ownership_revocation.authoritative_revocation_proof_ref(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id=worker_id,
        ownership_epoch=_BOUND_EPOCH,
    )
    assert proof is not None


def test_l3r8_cross_process_terminal_wins_before_revocation(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    worker, bound, worker_id, _owner = _running_with_worker(durable_bundle, repo)
    result = run_terminal_wins_before_revocation_race(
        db_path=durable_bundle.db_path,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=bound.revision,
    )
    assert len(result.outcomes) == 2
    assert all(code == 0 for code in result.exit_codes)
    assert "terminal" in result.outcomes
    assert "unauthorized" in result.outcomes
    loaded = durable_bundle.external_operation_state.load(_TENANT, _OP)
    assert loaded is not None
    assert loaded.physical_state is ExternalOperationPhysicalState.SUCCEEDED
    proof = durable_bundle.worker_execution_ownership_revocation.authoritative_revocation_proof_ref(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id=worker_id,
        ownership_epoch=_BOUND_EPOCH,
    )
    assert proof is None


def test_l3r8_revocation_without_supervision_fail_closed(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    _running_with_worker(durable_bundle, repo)
    reset_certified_owner_loss_composition_for_tests()
    configure_worker_execution_certified_owner_loss(
        worker_execution_ownership_revocation=durable_bundle.worker_execution_ownership_revocation,
        authoritative_state_store=durable_bundle.external_operation_state,
    )
    assert require_certified_owner_loss_composition().worker_execution_revocation_recorder is None
    with pytest.raises(WorkerExecutionOwnershipRevocationSupervisionNotConfiguredError):
        build_worker_execution_ownership_revocation_recorder(
            authoritative_state_store=durable_bundle.external_operation_state,
        )


def test_l3r8_forged_revocation_authorization_digest_rejected(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    _worker, bound, worker_id, _owner = _running_with_worker(durable_bundle, repo)
    forged = forged_worker_execution_ownership_revocation_authorization(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id=worker_id,
        ownership_epoch=_BOUND_EPOCH,
        bound_store_revision=bound.revision,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
    )
    from dataclasses import replace

    tampered = replace(forged, authorization_digest="0" * 64)
    with pytest.raises(WorkerExecutionDirectRevocationInvocationError):
        durable_bundle.worker_execution_ownership_revocation.revoke_worker_execution_ownership(
            tenant_id=_TENANT,
            operation_id=_OP,
            worker_instance_id=worker_id,
            ownership_epoch=_BOUND_EPOCH,
            authorization=tampered,
        )


def test_l3r8_in_transaction_fence_without_python_precheck(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    """Revocation row must block CAS inside SQLite even if pre-CAS read missed it."""
    from unittest.mock import patch

    from intergrax.contracts.certified_owner_loss_evidence import (
        CertifiedOwnerLossEvidenceRejectedError,
    )
    from intergrax.runtime.external_operations.external_operation_state_store import (
        ExternalOperationStateUpdate,
    )

    repo = InMemoryWorkerInstanceRepository()
    _running_with_worker(durable_bundle, repo)
    _revoke_bound(durable_bundle)
    store = durable_bundle.external_operation_state
    current = store.load(_TENANT, _OP)
    assert current is not None
    with patch(
        "intergrax.runtime.external_operations.external_operation_state_store."
        "bound_worker_execution_epoch_is_revoked",
        return_value=False,
    ):
        with pytest.raises(CertifiedOwnerLossEvidenceRejectedError):
            store.compare_and_set(
                _TENANT,
                _OP,
                expected_revision=current.revision,
                update=ExternalOperationStateUpdate(
                    physical_state=ExternalOperationPhysicalState.SUCCEEDED,
                ),
            )
