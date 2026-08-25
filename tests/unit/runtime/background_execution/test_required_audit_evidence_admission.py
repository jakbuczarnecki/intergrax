# © Artur Czarnecki. All rights reserved.

"""BG-EXEC-3 — required audit evidence admission semantics."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_event_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.background_execution.bootstrap import BackgroundExecutionIdentity
from intergrax.runtime.background_execution.identity_persistence import (
    KvBackgroundExecutionIdentityPersistence,
)
from intergrax.runtime.background_execution.required_audit_evidence import (
    RequiredAuditEvidencePersistenceError,
    admit_background_execution_handler,
    build_transport_triggered_execution_evidence,
    persist_required_audit_evidence,
)
from intergrax.runtime.background_execution.transport_ref import (
    BackgroundTransportExecutionRef,
)
from intergrax.runtime.nexus.errors.classifier import ErrorClassifier
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.observability.causal_evidence import CausalRelationKind
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
)
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)

pytestmark = pytest.mark.unit


class _KV:
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], bytes] = {}

    def get(self, tenant_id: str, key: str) -> bytes | None:
        return self._data.get((tenant_id, key))

    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        self._data[(tenant_id, key)] = value

    def delete(self, tenant_id: str, key: str) -> None:
        self._data.pop((tenant_id, key), None)

    def compare_and_set(
        self,
        tenant_id: str,
        key: str,
        expected: bytes | None,
        new_value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> bool:
        current = self.get(tenant_id, key)
        if expected is None and current is not None:
            return False
        if expected is not None and current != expected:
            return False
        self.set(tenant_id, key, new_value, ttl_seconds=ttl_seconds)
        return True


def _transport_ref() -> BackgroundTransportExecutionRef:
    return BackgroundTransportExecutionRef(
        tenant_id="tenant-a",
        provider="broker",
        transport_task_id="transport-1",
    )


def _identity(*, attempt_id: AttemptId | None = None) -> BackgroundExecutionIdentity:
    return BackgroundExecutionIdentity(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=attempt_id or mint_attempt_id(),
    )


def test_required_evidence_success_invokes_handler_once() -> None:
    persistence = InMemoryCausalEvidencePersistence()
    handler = Mock(return_value="ok")
    transport_ref = _transport_ref()
    execution_identity = _identity()

    result = admit_background_execution_handler(
        transport_ref=transport_ref,
        execution_identity=execution_identity,
        causal_evidence_persistence=persistence,
        handler=handler,
    )

    assert result == "ok"
    handler.assert_called_once()
    stored = persistence.list_for_execution(
        tenant_id=execution_identity.tenant_id,
        task_id=execution_identity.task_id,
        run_id=execution_identity.run_id,
    )
    assert len(stored) == 1
    assert stored[0].relation_kind == CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION


def test_required_evidence_backend_failure_blocks_handler_and_wraps_cause() -> None:
    backend_error = RuntimeError("database unavailable")
    persistence = Mock(spec=CausalEvidencePersistence)
    persistence.append.side_effect = backend_error
    handler = Mock()

    with pytest.raises(
        RequiredAuditEvidencePersistenceError,
        match="required audit evidence persistence failed",
    ) as exc_info:
        admit_background_execution_handler(
            transport_ref=_transport_ref(),
            execution_identity=_identity(),
            causal_evidence_persistence=persistence,
            handler=handler,
        )

    assert exc_info.value.__cause__ is backend_error
    handler.assert_not_called()
    assert (
        ErrorClassifier.classify(exc_info.value) == RuntimeErrorCode.DEPENDENCY_ERROR
    )


def test_already_typed_persistence_error_propagates_without_double_wrap() -> None:
    original = RequiredAuditEvidencePersistenceError("store down")
    persistence = Mock(spec=CausalEvidencePersistence)
    persistence.append.side_effect = original
    handler = Mock()

    with pytest.raises(RequiredAuditEvidencePersistenceError, match="store down") as exc_info:
        admit_background_execution_handler(
            transport_ref=_transport_ref(),
            execution_identity=_identity(),
            causal_evidence_persistence=persistence,
            handler=handler,
        )

    assert exc_info.value is original
    handler.assert_not_called()


def test_unsupported_relation_kind_raises_value_error_not_persistence_error() -> None:
    persistence = Mock(spec=CausalEvidencePersistence)
    evidence = Mock()
    evidence.relation_kind = "unsupported_relation"

    with pytest.raises(ValueError, match="not required audit evidence"):
        persist_required_audit_evidence(persistence, evidence)

    persistence.append.assert_not_called()


def test_required_evidence_failure_classified_as_dependency_error() -> None:
    code = ErrorClassifier.classify(
        RequiredAuditEvidencePersistenceError("persistence failed")
    )
    assert code == RuntimeErrorCode.DEPENDENCY_ERROR


def test_same_attempt_evidence_identity_stable_across_persistence_retry() -> None:
    persistence = InMemoryCausalEvidencePersistence()
    transport_ref = _transport_ref()
    execution_identity = _identity()
    stable_id: EventId = mint_event_id()
    evidence = build_transport_triggered_execution_evidence(
        transport_ref,
        execution_identity,
        evidence_id=stable_id,
    )

    persistence.append(evidence)
    persistence.append(evidence)

    stored = persistence.list_for_execution(
        tenant_id=execution_identity.tenant_id,
        task_id=execution_identity.task_id,
        run_id=execution_identity.run_id,
    )
    assert len(stored) == 1
    assert stored[0].evidence_id == stable_id


def test_worker_retry_mints_new_attempt_and_new_evidence() -> None:
    kv = _KV()
    identity_persistence = KvBackgroundExecutionIdentityPersistence(kv)
    persistence = InMemoryCausalEvidencePersistence()
    transport_ref = _transport_ref()
    handler_calls: list[AttemptId] = []

    def _handler_for(attempt_id: AttemptId) -> str:
        handler_calls.append(attempt_id)
        return "ok"

    from intergrax.runtime.background_execution.bootstrap import (
        bootstrap_background_execution,
    )

    first_identity = bootstrap_background_execution(
        transport_ref=transport_ref,
        identity_persistence=identity_persistence,
    )
    admit_background_execution_handler(
        transport_ref=transport_ref,
        execution_identity=first_identity,
        causal_evidence_persistence=persistence,
        handler=lambda: _handler_for(first_identity.attempt_id),
    )

    second_identity = bootstrap_background_execution(
        transport_ref=transport_ref,
        identity_persistence=identity_persistence,
    )
    admit_background_execution_handler(
        transport_ref=transport_ref,
        execution_identity=second_identity,
        causal_evidence_persistence=persistence,
        handler=lambda: _handler_for(second_identity.attempt_id),
    )

    assert first_identity.task_id == second_identity.task_id
    assert first_identity.run_id == second_identity.run_id
    assert first_identity.attempt_id != second_identity.attempt_id
    assert len(handler_calls) == 2

    stored = persistence.list_for_execution(
        tenant_id=first_identity.tenant_id,
        task_id=first_identity.task_id,
        run_id=first_identity.run_id,
    )
    assert len(stored) == 2
    assert stored[0].target.attempt_id == first_identity.attempt_id
    assert stored[1].target.attempt_id == second_identity.attempt_id
    assert stored[0].evidence_id != stored[1].evidence_id
