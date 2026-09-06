# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from intergrax.contracts.execution_identity import (
    EventId,
    mint_attempt_id,
    mint_event_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.diagnostics.diagnostic_scope_discovery_models import (
    DiagnosticScopeDiscoveryIntegrityError,
    DiagnosticScopeDiscoveryRequest,
    DiagnosticScopeDiscoveryStatus,
    DiagnosticScopeReferenceKind,
    EventScopeReference,
    ProblemScopeReference,
    TransportScopeReference,
)
from intergrax.runtime.diagnostics.diagnostic_scope_discovery_provider import (
    DiagnosticScopeProviderIntegrityError,
    DiagnosticScopeProviderUnavailableError,
    assert_diagnostic_scope_discovery_provider_conformance,
)
from intergrax.runtime.diagnostics.diagnostic_scope_discovery_service import (
    DiagnosticScopeDiscoveryService,
)
from intergrax.runtime.diagnostics.persistence_conformance import sample_problem
from intergrax.runtime.diagnostics.problem_grouping import problem_grouping_subject_ref_for_execution
from intergrax.runtime.diagnostics.providers.causal_transport_scope_provider import (
    CausalTransportScopeProvider,
)
from intergrax.runtime.diagnostics.providers.problem_scope_provider import ProblemScopeProvider
from intergrax.runtime.diagnostics.providers.runtime_event_scope_provider import (
    RUNTIME_EVENT_SCOPE_PROVIDER_ID,
    RuntimeEventScopeProvider,
)
from intergrax.runtime.events.execution_position import (
    ExecutionEventPosition,
    PositionedRuntimeEvent,
)
from intergrax.runtime.events.persistence_contract import (
    RuntimeEventPersistence,
    RuntimeEventPersistenceIntegrityError,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.causal_evidence import (
    CausalRelationKind,
    MessageBusTaskRef,
    PlatformCausalEvidence,
    RuntimeExecutionRef,
)
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    create_problem_for_tests,
    document_store_occurrence_persistence_for_tests,
    document_store_problem_persistence_for_tests,
    in_memory_document_store_for_problem_tests,
)

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_PROVIDER_FILE = Path(
    "intergrax/runtime/diagnostics/providers/runtime_event_scope_provider.py",
)

_FORBIDDEN_IMPORT_TOKENS = (
    "sqlite",
    "DocumentBackedRuntimeEventStore",
    "DocumentStore",
    "kafka",
    "celery",
    "rabbitmq",
    "worker",
    "platform_proofs",
    "qdrant",
    "pymongo",
)


@dataclass(frozen=True, slots=True)
class _FaultyRuntimeEventPersistence(RuntimeEventPersistence):
    positioned: PositionedRuntimeEvent | None
    mode: str = "ok"

    def append(self, event, *, tenant_id: str):
        raise AssertionError("append must not be called")

    def list_for_run(self, run_id, *, tenant_id: str, limit: int = 1000):
        raise AssertionError("list_for_run must not be called")

    def list_for_task(self, task_id, *, tenant_id: str, limit: int = 1000):
        raise AssertionError("list_for_task must not be called")

    def list_positioned_for_run(self, run_id, *, tenant_id: str, limit: int = 1000):
        raise AssertionError("list_positioned_for_run must not be called")

    def get_by_event_id(self, *, tenant_id: str, event_id: EventId):
        if self.mode == "integrity":
            raise RuntimeEventPersistenceIntegrityError("faulty persistence")
        if self.mode == "connection":
            raise ConnectionError("backend down")
        if self.mode == "timeout":
            raise TimeoutError("backend timeout")
        if self.mode == "value_error":
            raise ValueError("unexpected persistence failure")
        if self.mode == "mismatch" and self.positioned is not None:
            mismatched = self.positioned.event.model_copy(
                update={"event_id": mint_event_id()},
            )
            return PositionedRuntimeEvent(
                event=mismatched,
                position=self.positioned.position,
            )
        return self.positioned

    def close(self) -> None:
        return None


def _event_provider(
    persistence: RuntimeEventPersistence | None = None,
) -> RuntimeEventScopeProvider:
    return RuntimeEventScopeProvider(
        runtime_event_persistence=persistence or InMemoryRuntimeEventStore(),
    )


def _event_request(
    event_id: EventId,
    *,
    candidate_limit: int = 10,
) -> DiagnosticScopeDiscoveryRequest:
    return DiagnosticScopeDiscoveryRequest(
        tenant_id=_TENANT,
        reference=EventScopeReference(event_id=event_id),
        candidate_limit=candidate_limit,
    )


def _service(
    persistence: RuntimeEventPersistence | None = None,
) -> DiagnosticScopeDiscoveryService:
    return DiagnosticScopeDiscoveryService(providers=(_event_provider(persistence),))


def test_missing_event_id_returns_not_found() -> None:
    event_id = mint_event_id()
    result = _service(InMemoryRuntimeEventStore()).discover_scope(_event_request(event_id))
    assert result.status is DiagnosticScopeDiscoveryStatus.NOT_FOUND
    assert result.candidate_count == 0
    assert result.candidate_count_exact is True


def test_exact_event_returns_resolved() -> None:
    store = InMemoryRuntimeEventStore()
    event = sample_runtime_event(tenant_id=_TENANT)
    store.append(event, tenant_id=_TENANT)
    result = _service(store).discover_scope(_event_request(event.event_id))
    assert result.status is DiagnosticScopeDiscoveryStatus.RESOLVED
    assert result.resolved_scope is not None
    assert result.resolved_scope.task_id == event.task_id
    assert result.resolved_scope.run_id == event.run_id
    assert result.candidate_count == 1
    assert result.candidate_count_exact is True


def test_different_runtime_event_type_still_resolved() -> None:
    store = InMemoryRuntimeEventStore()
    event = sample_runtime_event(tenant_id=_TENANT).model_copy(
        update={"event_type": RuntimeEventType.TOOL_FAILED},
    )
    store.append(event, tenant_id=_TENANT)
    result = _service(store).discover_scope(_event_request(event.event_id))
    assert result.status is DiagnosticScopeDiscoveryStatus.RESOLVED


def test_event_tenant_none_with_explicit_persistence_tenant_resolves() -> None:
    store = InMemoryRuntimeEventStore()
    event = sample_runtime_event(tenant_id=None)
    store.append(event, tenant_id=_TENANT)
    result = _service(store).discover_scope(_event_request(event.event_id))
    assert result.status is DiagnosticScopeDiscoveryStatus.RESOLVED
    assert result.resolved_scope is not None
    assert result.resolved_scope.task_id == event.task_id
    assert result.resolved_scope.run_id == event.run_id


def test_wrong_tenant_lookup_returns_not_found() -> None:
    store = InMemoryRuntimeEventStore()
    event = sample_runtime_event(tenant_id=_TENANT)
    store.append(event, tenant_id=_TENANT)
    request = DiagnosticScopeDiscoveryRequest(
        tenant_id="tenant-b",
        reference=EventScopeReference(event_id=event.event_id),
    )
    result = _service(store).discover_scope(request)
    assert result.status is DiagnosticScopeDiscoveryStatus.NOT_FOUND


def test_event_id_mismatch_raises_integrity_error() -> None:
    event = sample_runtime_event(tenant_id=_TENANT)
    positioned = PositionedRuntimeEvent(
        event=event,
        position=ExecutionEventPosition(7),
    )
    persistence = _FaultyRuntimeEventPersistence(positioned=positioned, mode="mismatch")
    with pytest.raises(DiagnosticScopeDiscoveryIntegrityError, match="event_id"):
        _service(persistence).discover_scope(_event_request(event.event_id))


def test_persistence_integrity_maps_to_provider_integrity() -> None:
    persistence = _FaultyRuntimeEventPersistence(positioned=None, mode="integrity")
    with pytest.raises(DiagnosticScopeDiscoveryIntegrityError, match="faulty persistence"):
        _service(persistence).discover_scope(_event_request(mint_event_id()))


def test_connection_error_maps_to_provider_unavailable() -> None:
    persistence = _FaultyRuntimeEventPersistence(positioned=None, mode="connection")
    result = _service(persistence).discover_scope(_event_request(mint_event_id()))
    assert result.status is DiagnosticScopeDiscoveryStatus.PROVIDER_UNAVAILABLE


def test_timeout_error_maps_to_provider_unavailable() -> None:
    persistence = _FaultyRuntimeEventPersistence(positioned=None, mode="timeout")
    result = _service(persistence).discover_scope(_event_request(mint_event_id()))
    assert result.status is DiagnosticScopeDiscoveryStatus.PROVIDER_UNAVAILABLE


def test_unexpected_value_error_propagates() -> None:
    persistence = _FaultyRuntimeEventPersistence(positioned=None, mode="value_error")
    with pytest.raises(ValueError, match="unexpected persistence failure"):
        _service(persistence).discover_scope(_event_request(mint_event_id()))


def test_position_preservation_does_not_mutate_persistence() -> None:
    store = InMemoryRuntimeEventStore()
    event = sample_runtime_event(tenant_id=_TENANT)
    positioned = store.append(event, tenant_id=_TENANT)
    assert positioned.position == ExecutionEventPosition(1)
    _service(store).discover_scope(_event_request(event.event_id))
    lookup = store.get_by_event_id(tenant_id=_TENANT, event_id=event.event_id)
    assert lookup is not None
    assert lookup.position == positioned.position


def test_read_only_calls_get_by_event_id_once() -> None:
    persistence = MagicMock(spec=RuntimeEventPersistence)
    persistence.get_by_event_id.return_value = None
    _service(persistence).discover_scope(_event_request(mint_event_id()))
    persistence.get_by_event_id.assert_called_once()
    persistence.append.assert_not_called()
    persistence.list_for_run.assert_not_called()
    persistence.list_for_task.assert_not_called()
    persistence.list_positioned_for_run.assert_not_called()


def test_provider_has_no_forbidden_backend_imports() -> None:
    source = _PROVIDER_FILE.read_text(encoding="utf-8")
    violations = [token for token in _FORBIDDEN_IMPORT_TOKENS if token in source]
    assert not violations, f"forbidden imports: {violations}"


def test_provider_conformance() -> None:
    assert_diagnostic_scope_discovery_provider_conformance(
        _event_provider(),
        expected_provider_id=RUNTIME_EVENT_SCOPE_PROVIDER_ID,
        expected_reference_kind=DiagnosticScopeReferenceKind.EVENT,
    )


def test_deterministic_repeated_lookup() -> None:
    store = InMemoryRuntimeEventStore()
    event = sample_runtime_event(tenant_id=_TENANT)
    store.append(event, tenant_id=_TENANT)
    service = _service(store)
    request = _event_request(event.event_id)
    first = service.discover_scope(request)
    second = service.discover_scope(request)
    assert first == second


def test_candidate_limit_does_not_alter_exact_resolution() -> None:
    store = InMemoryRuntimeEventStore()
    event = sample_runtime_event(tenant_id=_TENANT)
    store.append(event, tenant_id=_TENANT)
    result = _service(store).discover_scope(
        _event_request(event.event_id, candidate_limit=1),
    )
    assert result.status is DiagnosticScopeDiscoveryStatus.RESOLVED
    assert result.candidate_count == 1


def test_in_memory_integration_resolves_task_run() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    event = sample_runtime_event(tenant_id=None, task_id=task_id, run_id=run_id)
    store.append(event, tenant_id=_TENANT)
    result = _service(store).discover_scope(_event_request(event.event_id))
    assert result.status is DiagnosticScopeDiscoveryStatus.RESOLVED
    assert result.resolved_scope is not None
    assert result.resolved_scope.task_id == task_id
    assert result.resolved_scope.run_id == run_id


def test_provenance_uses_runtime_event_canonical_ref() -> None:
    store = InMemoryRuntimeEventStore()
    event = sample_runtime_event(tenant_id=_TENANT)
    store.append(event, tenant_id=_TENANT)
    result = _service(store).discover_scope(_event_request(event.event_id))
    assert result.provenance[0].provider_id == RUNTIME_EVENT_SCOPE_PROVIDER_ID
    assert result.provenance[0].reference_kind is DiagnosticScopeReferenceKind.EVENT
    assert result.provenance[0].canonical_record_ref == f"runtime_event:{event.event_id}"


def test_three_provider_pluginability_through_unchanged_service() -> None:
    store = in_memory_document_store_for_problem_tests()
    problem_persistence = document_store_problem_persistence_for_tests(store)
    occurrence_persistence = document_store_occurrence_persistence_for_tests(store)
    causal_persistence = InMemoryCausalEvidencePersistence()
    runtime_store = InMemoryRuntimeEventStore()

    subject = problem_grouping_subject_ref_for_execution(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )
    problem = create_problem_for_tests(
        problem_persistence,
        sample_problem(tenant_id=_TENANT, subject_refs=(subject,), occurrence_count=0),
        indexed_subject_refs=(subject,),
    )

    transport_task_id = "celery-task-1"
    task_id = mint_task_id()
    run_id = mint_run_id()
    causal_persistence.append(
        PlatformCausalEvidence(
            evidence_id=mint_event_id(),
            relation_kind=CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION,
            tenant_id=_TENANT,
            source=MessageBusTaskRef(
                provider="celery",
                task_id=transport_task_id,
                tenant_id=_TENANT,
            ),
            target=RuntimeExecutionRef(
                task_id=task_id,
                run_id=run_id,
                attempt_id=mint_attempt_id(),
                tenant_id=_TENANT,
            ),
            recorded_at=sample_runtime_event().timestamp,
        ),
    )

    event = sample_runtime_event(tenant_id=None, task_id=mint_task_id(), run_id=mint_run_id())
    runtime_store.append(event, tenant_id=_TENANT)

    service = DiagnosticScopeDiscoveryService(
        providers=(
            ProblemScopeProvider(
                problem_persistence=problem_persistence,
                occurrence_persistence=occurrence_persistence,
            ),
            CausalTransportScopeProvider(causal_evidence_persistence=causal_persistence),
            RuntimeEventScopeProvider(runtime_event_persistence=runtime_store),
        ),
    )

    problem_result = service.discover_scope(
        DiagnosticScopeDiscoveryRequest(
            tenant_id=_TENANT,
            reference=ProblemScopeReference(problem_id=problem.problem_id),
        ),
    )
    assert problem_result.status is DiagnosticScopeDiscoveryStatus.INSUFFICIENT_EVIDENCE

    transport_result = service.discover_scope(
        DiagnosticScopeDiscoveryRequest(
            tenant_id=_TENANT,
            reference=TransportScopeReference(
                provider="celery",
                transport_task_id=transport_task_id,
            ),
        ),
    )
    assert transport_result.status is DiagnosticScopeDiscoveryStatus.RESOLVED
    assert transport_result.resolved_scope is not None
    assert transport_result.resolved_scope.task_id == task_id

    event_result = service.discover_scope(_event_request(event.event_id))
    assert event_result.status is DiagnosticScopeDiscoveryStatus.RESOLVED
    assert event_result.resolved_scope is not None
    assert event_result.resolved_scope.task_id == event.task_id
    assert event_result.resolved_scope.run_id == event.run_id
