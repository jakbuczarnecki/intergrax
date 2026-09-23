# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from intergrax.contracts.agent_governance_hitl import (
    LogicalInvocationFingerprint,
    digest_logical_invocation_fingerprint,
)
from intergrax.contracts.execution.suspended_operation.claim import (
    SuspendedOperationMutationOutcome,
)
from intergrax.contracts.execution.suspended_operation.persistence_conflict import (
    SuspendedOperationPersistenceConflictError,
)
from intergrax.contracts.execution.suspended_operation.codec import (
    SuspendedOperationKind,
)
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedExecutionOperationDescriptor,
    SuspendedOperationMaterializationState,
)
from intergrax.contracts.execution.suspended_operation.entity_id import (
    mint_suspended_operation_id,
)
from intergrax.contracts.execution.suspended_operation.payload_catalog import (
    CODE_EXEC_INPUT_SCHEMA_ID,
    ExecutionBoundCatalogToolOperationPayload,
)
from intergrax.contracts.execution.suspended_operation.authority_scope import (
    SuspendedOperationAuthorityScope,
)
from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.runtime.execution.suspended_operation.catalog_tool_codec import (
    ExecutionBoundCatalogToolPayloadCodec,
)
from intergrax.runtime.execution.suspended_operation.catalog_tool_invocation_intent import (
    digest_execution_bound_catalog_tool_invocation_intent,
)
from intergrax.runtime.execution.suspended_operation.document_store_suspended_operation_store import (
    DocumentStoreSuspendedExecutionOperationStore,
)
from intergrax.runtime.execution.suspended_operation.payload_digest import (
    digest_suspended_operation_envelope,
)
from intergrax.tools.providers.sandbox.contracts import CodeExecInput
from tests.unit.runtime.execution.suspended_operation.test_suspended_operation_store import (
    _identity,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[5]
HOST = (
    REPO
    / "intergrax"
    / "runtime"
    / "nexus"
    / "tools"
    / "continuation_aware_catalog_tool_host.py"
)
DOC_STORE = (
    REPO
    / "intergrax"
    / "runtime"
    / "execution"
    / "suspended_operation"
    / "document_store_suspended_operation_store.py"
)


def _base_payload(
    *, invocation_scope_id: str, step_id: str = "step-1"
) -> ExecutionBoundCatalogToolOperationPayload:
    return ExecutionBoundCatalogToolOperationPayload(
        tool_id="code.exec",
        tool_input_schema_id=CODE_EXEC_INPUT_SCHEMA_ID,
        tool_input=CodeExecInput(code="print(1)", language="python"),
        tenant_id="tenant-a",
        task_id=str(_identity().task_id),
        run_id=str(_identity().run_id),
        agent_id="agent-a",
        step_id=step_id,
        invocation_scope_id=invocation_scope_id,
        idempotency_key="idem-1",
    )


def _fingerprint_for_identity(
    *,
    invocation_intent_digest: str,
    step_id: str = "step-1",
    idempotency_key: str = "idem-1",
    identity=None,
) -> str:
    ident = identity or _identity()
    return digest_logical_invocation_fingerprint(
        task_id=str(ident.task_id),
        run_id=str(ident.run_id),
        attempt_id=str(ident.attempt_id),
        execution_id=str(ident.execution_id),
        tenant_id="tenant-a",
        agent_id="agent-a",
        tool_id="code.exec",
        step_id=step_id,
        idempotency_key=idempotency_key,
        invocation_intent_digest=invocation_intent_digest,
    ).digest


def _descriptor_with_fingerprint(
    fingerprint_digest: str,
    *,
    scope_id: str,
    continuation_id: str = "gcr_test",
) -> SuspendedExecutionOperationDescriptor:
    payload = _base_payload(invocation_scope_id=scope_id)
    codec = ExecutionBoundCatalogToolPayloadCodec()
    envelope = codec.encode(payload)
    return SuspendedExecutionOperationDescriptor(
        suspended_operation_id=mint_suspended_operation_id(),
        operation_kind=SuspendedOperationKind.EXECUTION_BOUND_CATALOG_TOOL,
        identity=_identity(),
        continuation_id=continuation_id,
        invocation_scope_id=scope_id,
        materialization_state=SuspendedOperationMaterializationState.PREPARED,
        materialization_revision=0,
        payload_digest=digest_suspended_operation_envelope(envelope),
        payload=envelope,
        logical_invocation_fingerprint=LogicalInvocationFingerprint(
            digest=fingerprint_digest,
        ),
        authority_scope=SuspendedOperationAuthorityScope.AGENT_RUNTIME_GOVERNANCE,
    )


def test_same_invocation_different_agr_scope_same_logical_fingerprint() -> None:
    payload_a = _base_payload(invocation_scope_id="agr_scope_a")
    payload_b = _base_payload(invocation_scope_id="agr_scope_b")
    intent_a = digest_execution_bound_catalog_tool_invocation_intent(payload_a)
    intent_b = digest_execution_bound_catalog_tool_invocation_intent(payload_b)
    assert intent_a == intent_b
    codec = ExecutionBoundCatalogToolPayloadCodec()
    digest_a = digest_suspended_operation_envelope(codec.encode(payload_a))
    digest_b = digest_suspended_operation_envelope(codec.encode(payload_b))
    assert digest_a != digest_b
    assert _fingerprint_for_identity(
        invocation_intent_digest=intent_a
    ) == _fingerprint_for_identity(
        invocation_intent_digest=intent_b,
    )


def test_different_tool_input_different_fingerprint() -> None:
    base = _base_payload(invocation_scope_id="agr_x")
    other = base.model_copy(
        update={
            "tool_input": CodeExecInput(code="print(2)", language="python"),
        },
    )
    intent_base = digest_execution_bound_catalog_tool_invocation_intent(base)
    intent_other = digest_execution_bound_catalog_tool_invocation_intent(other)
    assert _fingerprint_for_identity(
        invocation_intent_digest=intent_base
    ) != _fingerprint_for_identity(
        invocation_intent_digest=intent_other,
    )


def test_different_idempotency_key_different_fingerprint() -> None:
    intent = digest_execution_bound_catalog_tool_invocation_intent(
        _base_payload(invocation_scope_id="agr_x"),
    )
    assert _fingerprint_for_identity(
        invocation_intent_digest=intent, idempotency_key="a"
    ) != (
        _fingerprint_for_identity(invocation_intent_digest=intent, idempotency_key="b")
    )


def test_different_step_different_fingerprint() -> None:
    intent = digest_execution_bound_catalog_tool_invocation_intent(
        _base_payload(invocation_scope_id="agr_x"),
    )
    assert _fingerprint_for_identity(invocation_intent_digest=intent, step_id="s1") != (
        _fingerprint_for_identity(invocation_intent_digest=intent, step_id="s2")
    )


def test_different_four_id_different_fingerprint() -> None:
    from intergrax.contracts.execution_continuation import ExecutionContinuationIdentity
    from intergrax.contracts.execution_identity import (
        mint_attempt_id,
        mint_execution_id,
        mint_run_id,
        mint_task_id,
    )

    intent = digest_execution_bound_catalog_tool_invocation_intent(
        _base_payload(invocation_scope_id="agr_x"),
    )
    other_identity = ExecutionContinuationIdentity(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    assert _fingerprint_for_identity(
        invocation_intent_digest=intent
    ) != _fingerprint_for_identity(
        invocation_intent_digest=intent,
        identity=other_identity,
    )


def test_two_store_instances_same_fingerprint_one_active_descriptor() -> None:
    document_store = InMemoryDocumentStore()
    store_a = DocumentStoreSuspendedExecutionOperationStore(document_store)
    store_b = DocumentStoreSuspendedExecutionOperationStore(document_store)
    intent = digest_execution_bound_catalog_tool_invocation_intent(
        _base_payload(invocation_scope_id="agr_a"),
    )
    fp = _fingerprint_for_identity(invocation_intent_digest=intent)
    desc_a = _descriptor_with_fingerprint(fp, scope_id="agr_a", continuation_id="gcr_a")
    desc_b = _descriptor_with_fingerprint(fp, scope_id="agr_b", continuation_id="gcr_b")
    barrier = threading.Barrier(2)
    outcomes: list = []
    errors: list[BaseException] = []

    def _prepare(store: DocumentStoreSuspendedExecutionOperationStore, desc):
        try:
            barrier.wait()
            try:
                outcomes.append(store.prepare(desc))
            except SuspendedOperationPersistenceConflictError:
                outcomes.append(store.prepare(desc))
        except BaseException as exc:
            errors.append(exc)

    t_a = threading.Thread(target=_prepare, args=(store_a, desc_a))
    t_b = threading.Thread(target=_prepare, args=(store_b, desc_b))
    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()
    assert not errors
    applied = [
        item
        for item in outcomes
        if item.outcome is SuspendedOperationMutationOutcome.APPLIED
    ]
    already = [
        item
        for item in outcomes
        if item.outcome is SuspendedOperationMutationOutcome.ALREADY_ACTIVE
    ]
    assert len(applied) == 1
    assert len(already) == 1
    winner = applied[0].descriptor
    assert winner is not None
    assert already[0].descriptor is not None
    assert already[0].descriptor.suspended_operation_id == winner.suspended_operation_id
    reloaded = store_b.load_active_for_logical_invocation(
        winner.logical_invocation_fingerprint,
    )
    assert reloaded is not None
    assert reloaded.suspended_operation_id == winner.suspended_operation_id


def test_two_store_instances_different_fingerprints_both_persist() -> None:
    document_store = InMemoryDocumentStore()
    store_a = DocumentStoreSuspendedExecutionOperationStore(document_store)
    store_b = DocumentStoreSuspendedExecutionOperationStore(document_store)
    intent_a = digest_execution_bound_catalog_tool_invocation_intent(
        _base_payload(invocation_scope_id="agr_1"),
    )
    intent_b = digest_execution_bound_catalog_tool_invocation_intent(
        _base_payload(invocation_scope_id="agr_2", step_id="step-2"),
    )
    fp_a = _fingerprint_for_identity(invocation_intent_digest=intent_a)
    fp_b = _fingerprint_for_identity(
        invocation_intent_digest=intent_b,
        step_id="step-2",
    )
    desc_a = _descriptor_with_fingerprint(fp_a, scope_id="agr_1")
    desc_b = _descriptor_with_fingerprint(
        fp_b, scope_id="agr_2", continuation_id="gcr_b"
    )
    assert store_a.prepare(desc_a).outcome is SuspendedOperationMutationOutcome.APPLIED
    assert store_b.prepare(desc_b).outcome is SuspendedOperationMutationOutcome.APPLIED
    assert store_a.load(desc_a.suspended_operation_id) is not None
    assert store_b.load(desc_b.suspended_operation_id) is not None


def test_failed_cas_does_not_leave_ghost_active_descriptor() -> None:
    document_store = InMemoryDocumentStore()
    store_a = DocumentStoreSuspendedExecutionOperationStore(document_store)
    store_b = DocumentStoreSuspendedExecutionOperationStore(document_store)
    intent = digest_execution_bound_catalog_tool_invocation_intent(
        _base_payload(invocation_scope_id="agr_win"),
    )
    fp = _fingerprint_for_identity(invocation_intent_digest=intent)
    winner = _descriptor_with_fingerprint(fp, scope_id="agr_win")
    loser = _descriptor_with_fingerprint(
        fp, scope_id="agr_lose", continuation_id="gcr_lose"
    )
    assert store_a.prepare(winner).outcome is SuspendedOperationMutationOutcome.APPLIED
    try:
        store_b.prepare(loser)
    except SuspendedOperationPersistenceConflictError:
        retry = store_b.prepare(loser)
        assert retry.outcome is SuspendedOperationMutationOutcome.ALREADY_ACTIVE
    else:
        retry = store_b.prepare(loser)
        assert retry.outcome is SuspendedOperationMutationOutcome.ALREADY_ACTIVE
    active = store_a.load_active_for_logical_invocation(
        winner.logical_invocation_fingerprint,
    )
    assert active is not None
    assert active.suspended_operation_id == winner.suspended_operation_id


def test_host_gate_no_agr_probe() -> None:
    assert "agr_probe" not in HOST.read_text(encoding="utf-8")


def test_document_store_gate_snapshot_coupled_cas() -> None:
    source = DOC_STORE.read_text(encoding="utf-8")
    assert "expected_record = self._document_store.get" in source
    assert "_backing_from_record(expected_record)" in source
    assert "_persist_snapshot(backing, expected_record=expected_record)" in source
    assert "existing = self._document_store.get" not in source


def test_host_gate_no_durable_persist_message_routing() -> None:
    source = HOST.read_text(encoding="utf-8")
    assert "durable persist stale" not in source
    assert "durable persist race" not in source
    assert "SuspendedOperationPersistenceConflictError" in source
