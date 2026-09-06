# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.runtime.diagnostics.diagnostic_scope_discovery_models import (
    DiagnosticScopeDiscoveryConfigurationError,
    DiagnosticScopeDiscoveryIntegrityError,
    DiagnosticScopeDiscoveryRequest,
    DiagnosticScopeDiscoveryStatus,
    DiagnosticScopeReferenceKind,
    EventScopeReference,
    ProblemScopeReference,
    TransportScopeReference,
    unsupported_reference_result,
)
from intergrax.runtime.diagnostics.providers.problem_scope_provider import (
    PROBLEM_SCOPE_PROVIDER_ID,
    ProblemScopeProvider,
)
from intergrax.runtime.diagnostics.diagnostic_scope_discovery_provider import (
    DiagnosticScopeDiscoveryProviderRegistry,
    DiagnosticScopeProviderIntegrityError,
    DiagnosticScopeProviderResult,
    DiagnosticScopeProviderUnavailableError,
    assert_diagnostic_scope_discovery_provider_conformance,
)
from intergrax.runtime.diagnostics.diagnostic_scope_discovery_service import (
    DiagnosticScopeDiscoveryService,
)
from intergrax.runtime.diagnostics.problem_lifecycle import mint_problem_id
from intergrax.contracts.execution_identity import mint_event_id
from intergrax.runtime.diagnostics.providers.causal_transport_scope_provider import (
    CAUSAL_TRANSPORT_SCOPE_PROVIDER_ID,
    CausalTransportScopeProvider,
)
from intergrax.runtime.diagnostics.providers.runtime_event_scope_provider import (
    RUNTIME_EVENT_SCOPE_PROVIDER_ID,
    RuntimeEventScopeProvider,
)
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    document_store_occurrence_persistence_for_tests,
    document_store_problem_persistence_for_tests,
    in_memory_document_store_for_problem_tests,
)

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"


@dataclass(frozen=True, slots=True)
class _StubProvider:
    provider_id: str
    supported_reference_kind: DiagnosticScopeReferenceKind

    def discover(
        self,
        *,
        tenant_id: str,
        reference: ProblemScopeReference,
        candidate_limit: int,
    ) -> DiagnosticScopeProviderResult:
        return DiagnosticScopeProviderResult(
            status=DiagnosticScopeDiscoveryStatus.NOT_FOUND,
            resolved_scope=None,
            candidates=(),
            candidate_count=0,
            candidate_count_exact=True,
            provenance=(),
        )


def test_registry_rejects_duplicate_provider_ids() -> None:
    provider = _StubProvider("dup", DiagnosticScopeReferenceKind.PROBLEM)
    with pytest.raises(DiagnosticScopeDiscoveryConfigurationError, match="duplicate provider_id"):
        DiagnosticScopeDiscoveryProviderRegistry((provider, provider))


def test_registry_rejects_duplicate_reference_kinds() -> None:
    first = _StubProvider("one", DiagnosticScopeReferenceKind.PROBLEM)
    second = _StubProvider("two", DiagnosticScopeReferenceKind.PROBLEM)
    with pytest.raises(
        DiagnosticScopeDiscoveryConfigurationError,
        match="duplicate supported reference kind",
    ):
        DiagnosticScopeDiscoveryProviderRegistry((first, second))


def test_registry_resolves_provider_deterministically() -> None:
    provider = _StubProvider("problem_scope", DiagnosticScopeReferenceKind.PROBLEM)
    registry = DiagnosticScopeDiscoveryProviderRegistry((provider,))
    reference = ProblemScopeReference(problem_id=mint_problem_id())
    resolved = registry.resolve_for_reference(reference)
    assert resolved is provider


def test_registry_unsupported_reference_returns_none() -> None:
    registry = DiagnosticScopeDiscoveryProviderRegistry(())
    reference = ProblemScopeReference(problem_id=mint_problem_id())
    assert registry.resolve_for_reference(reference) is None
    assert unsupported_reference_result().status is DiagnosticScopeDiscoveryStatus.UNSUPPORTED_REFERENCE


def test_provider_conformance_helper() -> None:
    provider = _StubProvider("problem_scope", DiagnosticScopeReferenceKind.PROBLEM)
    assert_diagnostic_scope_discovery_provider_conformance(
        provider,
        expected_provider_id="problem_scope",
        expected_reference_kind=DiagnosticScopeReferenceKind.PROBLEM,
    )


@dataclass(frozen=True, slots=True)
class _SyntheticPluginProvider:
    provider_id: str = "synthetic_scope"
    supported_reference_kind: DiagnosticScopeReferenceKind = DiagnosticScopeReferenceKind.PROBLEM
    mode: str = "not_found"

    def discover(
        self,
        *,
        tenant_id: str,
        reference: ProblemScopeReference,
        candidate_limit: int,
    ) -> DiagnosticScopeProviderResult:
        if self.mode == "integrity":
            raise DiagnosticScopeProviderIntegrityError("synthetic integrity failure")
        if self.mode == "unavailable":
            raise DiagnosticScopeProviderUnavailableError("synthetic unavailable")
        return DiagnosticScopeProviderResult(
            status=DiagnosticScopeDiscoveryStatus.NOT_FOUND,
            resolved_scope=None,
            candidates=(),
            candidate_count=0,
            candidate_count_exact=True,
            provenance=(),
        )


def test_service_accepts_synthetic_provider_without_core_modification() -> None:
    provider = _SyntheticPluginProvider()
    service = DiagnosticScopeDiscoveryService(providers=(provider,))
    request = DiagnosticScopeDiscoveryRequest(
        tenant_id=_TENANT,
        reference=ProblemScopeReference(problem_id=mint_problem_id()),
    )
    result = service.discover_scope(request)
    assert result.status is DiagnosticScopeDiscoveryStatus.NOT_FOUND


def test_service_maps_synthetic_provider_integrity_error() -> None:
    provider = _SyntheticPluginProvider(mode="integrity")
    service = DiagnosticScopeDiscoveryService(providers=(provider,))
    request = DiagnosticScopeDiscoveryRequest(
        tenant_id=_TENANT,
        reference=ProblemScopeReference(problem_id=mint_problem_id()),
    )
    with pytest.raises(DiagnosticScopeDiscoveryIntegrityError, match="synthetic integrity"):
        service.discover_scope(request)


def test_service_maps_synthetic_provider_unavailable_error() -> None:
    provider = _SyntheticPluginProvider(mode="unavailable")
    service = DiagnosticScopeDiscoveryService(providers=(provider,))
    request = DiagnosticScopeDiscoveryRequest(
        tenant_id=_TENANT,
        reference=ProblemScopeReference(problem_id=mint_problem_id()),
    )
    result = service.discover_scope(request)
    assert result.status is DiagnosticScopeDiscoveryStatus.PROVIDER_UNAVAILABLE


def _production_providers() -> tuple[
    ProblemScopeProvider,
    CausalTransportScopeProvider,
    RuntimeEventScopeProvider,
]:
    store = in_memory_document_store_for_problem_tests()
    return (
        ProblemScopeProvider(
            problem_persistence=document_store_problem_persistence_for_tests(store),
            occurrence_persistence=document_store_occurrence_persistence_for_tests(store),
        ),
        CausalTransportScopeProvider(
            causal_evidence_persistence=InMemoryCausalEvidencePersistence(),
        ),
        RuntimeEventScopeProvider(
            runtime_event_persistence=InMemoryRuntimeEventStore(),
        ),
    )


def test_registry_resolves_problem_scope_provider() -> None:
    problem_provider, transport_provider, event_provider = _production_providers()
    registry = DiagnosticScopeDiscoveryProviderRegistry(
        (problem_provider, transport_provider, event_provider),
    )
    reference = ProblemScopeReference(problem_id=mint_problem_id())
    assert registry.resolve_for_reference(reference) is problem_provider


def test_registry_resolves_causal_transport_scope_provider() -> None:
    problem_provider, transport_provider, event_provider = _production_providers()
    registry = DiagnosticScopeDiscoveryProviderRegistry(
        (problem_provider, transport_provider, event_provider),
    )
    reference = TransportScopeReference(provider="celery", transport_task_id="task-1")
    assert registry.resolve_for_reference(reference) is transport_provider


def test_registry_resolves_runtime_event_scope_provider() -> None:
    problem_provider, transport_provider, event_provider = _production_providers()
    registry = DiagnosticScopeDiscoveryProviderRegistry(
        (problem_provider, transport_provider, event_provider),
    )
    reference = EventScopeReference(event_id=mint_event_id())
    assert registry.resolve_for_reference(reference) is event_provider


def test_problem_scope_provider_conformance() -> None:
    problem_provider, _, _ = _production_providers()
    assert_diagnostic_scope_discovery_provider_conformance(
        problem_provider,
        expected_provider_id=PROBLEM_SCOPE_PROVIDER_ID,
        expected_reference_kind=DiagnosticScopeReferenceKind.PROBLEM,
    )


def test_causal_transport_scope_provider_conformance() -> None:
    _, transport_provider, _ = _production_providers()
    assert_diagnostic_scope_discovery_provider_conformance(
        transport_provider,
        expected_provider_id=CAUSAL_TRANSPORT_SCOPE_PROVIDER_ID,
        expected_reference_kind=DiagnosticScopeReferenceKind.TRANSPORT,
    )


def test_runtime_event_scope_provider_conformance() -> None:
    _, _, event_provider = _production_providers()
    assert_diagnostic_scope_discovery_provider_conformance(
        event_provider,
        expected_provider_id=RUNTIME_EVENT_SCOPE_PROVIDER_ID,
        expected_reference_kind=DiagnosticScopeReferenceKind.EVENT,
    )
