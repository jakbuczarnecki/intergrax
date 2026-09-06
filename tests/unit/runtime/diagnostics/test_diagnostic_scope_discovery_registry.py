# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.runtime.diagnostics.diagnostic_scope_discovery_models import (
    DiagnosticScopeDiscoveryConfigurationError,
    DiagnosticScopeDiscoveryStatus,
    DiagnosticScopeReferenceKind,
    ProblemScopeReference,
    unsupported_reference_result,
)
from intergrax.runtime.diagnostics.diagnostic_scope_discovery_provider import (
    DiagnosticScopeDiscoveryProviderRegistry,
    DiagnosticScopeProviderResult,
    assert_diagnostic_scope_discovery_provider_conformance,
)
from intergrax.runtime.diagnostics.problem_lifecycle import mint_problem_id

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
