# © Artur Czarnecki. All rights reserved.

"""F8 redrive dispatch identity — adversarial admission and invoker negatives."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intergrax.contracts.external_operation_redrive_effect_binding import (
    ExternalOperationRedriveEffectBindingMismatchError,
)
from intergrax.contracts.external_operation_redrive_trusted_effect_contracts import (
    TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT,
)
from intergrax.runtime.external_operations.redrive_external_operation_authority import (
    bind_authoritative_redrive_effect_to_operation,
    transfer_redrive_authority_after_reconciliation,
)
from intergrax.runtime.external_operations.sanctioned_physical_redrive_dispatch_admission import (
    SanctionedPhysicalRedriveDispatchRejectedError,
    assert_sanctioned_physical_redrive_dispatch_admitted,
)
from intergrax.runtime.external_operations.sqlite_tool_life_durable_stores import (
    SqliteToolLifeDurableStoresBundle,
)
from intergrax.autonomous_work.in_memory_repository import InMemoryWorkerInstanceRepository
from intergrax.contracts.certified_owner_loss_evidence import CertifiedOwnerLossReason
from intergrax.runtime.external_operations.certified_owner_loss_composition import (
    reset_certified_owner_loss_composition_for_tests,
)
from intergrax.runtime.external_operations.external_operation_ownership import (
    ProcessLocalExternalOperationOwner,
    mark_physical_running,
)
from intergrax.runtime.external_operations.worker_execution_certified_owner_loss import (
    bind_worker_execution_ownership,
    build_worker_execution_certified_owner_loss_certifier,
    configure_worker_execution_certified_owner_loss,
)
from intergrax.runtime.tools.operation_identity import compute_invocation_operation_identity
from intergrax.tools.providers.http.contracts import HttpRequestInput
from intergrax.tools.providers.http.idempotent_post_qual import HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID
from intergrax.tools.providers.http.qualification_composition import (
    register_tool_life_http_class_b_qualification_tools,
)
from intergrax.tools.providers.http.service import HTTP_REQUEST_TOOL_ID
from intergrax.tools.providers.http.bundle import register_http_tools
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from tests.unit.autonomous_work.repository_contracts import worker_instance
from tests.unit.runtime.external_operations.test_tool_life_x_l3_r8_sqlite_durable_revocation_shared_store import (
    _BOUND_EPOCH,
    _TENANT,
    _revoke_bound,
)
from tests.unit.runtime.external_operations.tool_life_x_qualification_revocation_supervision import (
    QualificationWorkerExecutionRevocationSupervision,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_MAC_ENV = "INTERGRAX_WORKER_EXECUTION_REVOCATION_MAC_KEY"


@pytest.fixture(autouse=True)
def _tool_life_revocation_mac(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_MAC_ENV, "test-mac-key-redrive-dispatch-adv")
    reset_certified_owner_loss_composition_for_tests()
    yield
    reset_certified_owner_loss_composition_for_tests()

_IDEM = "redrive-adversarial-idem"
_BODY = json.dumps({"n": 1})
_HTTP_INPUT = HttpRequestInput(
    url="http://127.0.0.1:9/effects",
    method="POST",
    body=_BODY,
    timeout_s=1.0,
)
_FINGERPRINT = compute_invocation_operation_identity(
    HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID,
    _HTTP_INPUT,
).operation_fingerprint


@pytest.fixture
def durable_bundle(tmp_path: Path) -> SqliteToolLifeDurableStoresBundle:
    return SqliteToolLifeDurableStoresBundle(db_path=tmp_path / "redrive-dispatch.db")


@pytest.fixture
def qual_registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_http_tools(registry, ToolWiringContext())
    register_tool_life_http_class_b_qualification_tools(registry, ToolWiringContext())
    return registry


def _bind_class_b_and_transfer(
    bundle: SqliteToolLifeDurableStoresBundle,
    *,
    op_id: str,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    configure_worker_execution_certified_owner_loss(
        worker_execution_ownership_revocation=bundle.worker_execution_ownership_revocation,
        worker_instance_repository=repo,
        authoritative_state_store=bundle.external_operation_state,
        worker_execution_revocation_supervision=QualificationWorkerExecutionRevocationSupervision(),
    )
    store = bundle.external_operation_state
    worker = worker_instance()
    repo.create(worker)
    owner = ProcessLocalExternalOperationOwner.mint()
    record = store.create_if_absent(_TENANT, op_id, owner_token=owner.token)
    record = bind_authoritative_redrive_effect_to_operation(
        store,
        tenant_id=_TENANT,
        operation_id=op_id,
        expected_revision=record.revision,
        trusted_effect_contract_id=TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT.contract_id,
        provider_id="http_rest",
        tool_id="http_idempotent_post",
        physical_idempotency_key=_IDEM,
        physical_effect_fingerprint=_FINGERPRINT,
    )
    record = mark_physical_running(
        store,
        tenant_id=_TENANT,
        operation_id=op_id,
        expected_revision=record.revision,
        owner=owner,
    )
    bind_worker_execution_ownership(
        store,
        tenant_id=_TENANT,
        operation_id=op_id,
        expected_revision=record.revision,
        worker_instance_id=str(worker.worker_instance_id),
        worker_ownership_epoch=_BOUND_EPOCH,
    )
    proof = _revoke_bound(bundle, operation_id=op_id)
    certifier = build_worker_execution_certified_owner_loss_certifier(store=store)
    staged = certifier.stage_certified_owner_loss(
        tenant_id=_TENANT,
        operation_id=op_id,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
        liveness_proof_ref=proof,
    )
    unknown = store.apply_certified_owner_loss(
        _TENANT,
        op_id,
        expected_revision=staged.revision,
        evidence_id=staged.staged_certified_owner_loss.evidence.evidence_id,
    )
    transfer_redrive_authority_after_reconciliation(
        store,
        tenant_id=_TENANT,
        operation_id=op_id,
        expected_revision=unknown.revision,
        trusted_effect_contract=TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT,
        worker_instance_id="recovery-worker-1",
        worker_ownership_epoch=_BOUND_EPOCH + 1,
        provider_id="http_rest",
        tool_id="http_idempotent_post",
        physical_idempotency_key=_IDEM,
        physical_effect_fingerprint=_FINGERPRINT,
    )


def test_valid_class_b_dispatch_admission(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
    qual_registry: ToolRegistry,
) -> None:
    op_id = "op-valid-redrive"
    _bind_class_b_and_transfer(durable_bundle, op_id=op_id)
    registered = qual_registry.get(HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID)
    assert_sanctioned_physical_redrive_dispatch_admitted(
        durable_bundle.external_operation_state,
        tenant_id=_TENANT,
        durable_operation_id=op_id,
        catalog_tool_id=HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID,
        registered_tool=registered,
        physical_idempotency_key=_IDEM,
        tool_input=_HTTP_INPUT,
        physical_attempt_sequence=2,
    )


def test_wrong_catalog_tool_class_c_rejected(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
    qual_registry: ToolRegistry,
) -> None:
    op_id = "op-wrong-catalog"
    _bind_class_b_and_transfer(durable_bundle, op_id=op_id)
    with pytest.raises(ExternalOperationRedriveEffectBindingMismatchError):
        assert_sanctioned_physical_redrive_dispatch_admitted(
            durable_bundle.external_operation_state,
            tenant_id=_TENANT,
            durable_operation_id=op_id,
            catalog_tool_id=HTTP_REQUEST_TOOL_ID,
            registered_tool=qual_registry.get(HTTP_REQUEST_TOOL_ID),
            physical_idempotency_key=_IDEM,
            tool_input=_HTTP_INPUT,
            physical_attempt_sequence=2,
        )


def test_wrong_idempotency_key_rejected(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
    qual_registry: ToolRegistry,
) -> None:
    op_id = "op-wrong-idem"
    _bind_class_b_and_transfer(durable_bundle, op_id=op_id)
    with pytest.raises(ExternalOperationRedriveEffectBindingMismatchError):
        assert_sanctioned_physical_redrive_dispatch_admitted(
            durable_bundle.external_operation_state,
            tenant_id=_TENANT,
            durable_operation_id=op_id,
            catalog_tool_id=HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID,
            registered_tool=qual_registry.get(HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID),
            physical_idempotency_key="other-key",
            tool_input=_HTTP_INPUT,
            physical_attempt_sequence=2,
        )


def test_wrong_fingerprint_rejected(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
    qual_registry: ToolRegistry,
) -> None:
    op_id = "op-wrong-fp"
    _bind_class_b_and_transfer(durable_bundle, op_id=op_id)
    other_input = HttpRequestInput(
        url="http://127.0.0.1:9/effects",
        method="POST",
        body=json.dumps({"n": 2}),
        timeout_s=1.0,
    )
    with pytest.raises(ExternalOperationRedriveEffectBindingMismatchError):
        assert_sanctioned_physical_redrive_dispatch_admitted(
            durable_bundle.external_operation_state,
            tenant_id=_TENANT,
            durable_operation_id=op_id,
            catalog_tool_id=HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID,
            registered_tool=qual_registry.get(HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID),
            physical_idempotency_key=_IDEM,
            tool_input=other_input,
            physical_attempt_sequence=2,
        )


def test_wrong_operation_id_rejected(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
    qual_registry: ToolRegistry,
) -> None:
    op_id = "op-real"
    _bind_class_b_and_transfer(durable_bundle, op_id=op_id)
    with pytest.raises(SanctionedPhysicalRedriveDispatchRejectedError):
        assert_sanctioned_physical_redrive_dispatch_admitted(
            durable_bundle.external_operation_state,
            tenant_id=_TENANT,
            durable_operation_id="op-other",
            catalog_tool_id=HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID,
            registered_tool=qual_registry.get(HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID),
            physical_idempotency_key=_IDEM,
            tool_input=_HTTP_INPUT,
            physical_attempt_sequence=2,
        )


def test_without_redrive_authority_rejected(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
    qual_registry: ToolRegistry,
) -> None:
    op_id = "op-no-redrive"
    store = durable_bundle.external_operation_state
    record = store.create_if_absent(_TENANT, op_id)
    bind_authoritative_redrive_effect_to_operation(
        store,
        tenant_id=_TENANT,
        operation_id=op_id,
        expected_revision=record.revision,
        trusted_effect_contract_id=TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT.contract_id,
        provider_id="http_rest",
        tool_id="http_idempotent_post",
        physical_idempotency_key=_IDEM,
        physical_effect_fingerprint=_FINGERPRINT,
    )
    with pytest.raises(SanctionedPhysicalRedriveDispatchRejectedError):
        assert_sanctioned_physical_redrive_dispatch_admitted(
            store,
            tenant_id=_TENANT,
            durable_operation_id=op_id,
            catalog_tool_id=HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID,
            registered_tool=qual_registry.get(HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID),
            physical_idempotency_key=_IDEM,
            tool_input=_HTTP_INPUT,
            physical_attempt_sequence=2,
        )
