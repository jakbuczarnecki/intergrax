# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Provider contract and registry for diagnostic scope discovery (DG-002)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.runtime.diagnostics.diagnostic_scope_discovery_models import (
    DiagnosticExecutionScopeCandidate,
    DiagnosticScopeDiscoveryConfigurationError,
    DiagnosticScopeDiscoveryStatus,
    DiagnosticScopeReference,
    DiagnosticScopeReferenceKind,
    DiagnosticScopeResolutionProvenance,
)
from intergrax.runtime.diagnostics.diagnostic_subject import ExecutionDiagnosticSubjectRef


@dataclass(frozen=True, slots=True)
class DiagnosticScopeProviderResult:
    """Normalized provider output before public service projection."""

    status: DiagnosticScopeDiscoveryStatus
    resolved_scope: ExecutionDiagnosticSubjectRef | None
    candidates: tuple[DiagnosticExecutionScopeCandidate, ...]
    candidate_count: int
    candidate_count_exact: bool
    provenance: tuple[DiagnosticScopeResolutionProvenance, ...]
    limitations: tuple[str, ...] = ()


@runtime_checkable
class DiagnosticScopeDiscoveryProvider(Protocol):
    """Read-only scope discovery provider for one reference family."""

    @property
    def provider_id(self) -> str:
        """Stable provider identity."""

    @property
    def supported_reference_kind(self) -> DiagnosticScopeReferenceKind:
        """Reference discriminator handled by this provider."""

    def discover(
        self,
        *,
        tenant_id: str,
        reference: DiagnosticScopeReference,
        candidate_limit: int,
    ) -> DiagnosticScopeProviderResult:
        """Resolve execution scope candidates for one tenant-scoped reference."""


def assert_diagnostic_scope_discovery_provider_conformance(
    provider: DiagnosticScopeDiscoveryProvider,
    *,
    expected_provider_id: str,
    expected_reference_kind: DiagnosticScopeReferenceKind,
) -> None:
    """Validate generic provider contract semantics."""
    if provider.provider_id != expected_provider_id:
        raise AssertionError(
            f"provider_id mismatch: expected {expected_provider_id!r}, "
            f"got {provider.provider_id!r}",
        )
    if provider.supported_reference_kind is not expected_reference_kind:
        raise AssertionError(
            "supported_reference_kind mismatch: "
            f"expected {expected_reference_kind!r}, "
            f"got {provider.supported_reference_kind!r}",
        )
    if provider.provider_id != provider.provider_id.strip():
        raise AssertionError("provider_id must be stable trimmed identifier")


class DiagnosticScopeDiscoveryProviderRegistry:
    """Explicit typed provider registry with deterministic resolution order."""

    def __init__(
        self,
        providers: tuple[DiagnosticScopeDiscoveryProvider, ...],
    ) -> None:
        provider_ids: set[str] = set()
        reference_kinds: set[DiagnosticScopeReferenceKind] = set()
        ordered: list[DiagnosticScopeDiscoveryProvider] = []
        for provider in providers:
            provider_id = provider.provider_id
            reference_kind = provider.supported_reference_kind
            if provider_id in provider_ids:
                raise DiagnosticScopeDiscoveryConfigurationError(
                    f"duplicate provider_id: {provider_id!r}",
                )
            if reference_kind in reference_kinds:
                raise DiagnosticScopeDiscoveryConfigurationError(
                    f"duplicate supported reference kind: {reference_kind.value!r}",
                )
            provider_ids.add(provider_id)
            reference_kinds.add(reference_kind)
            ordered.append(provider)
        self._providers = tuple(ordered)

    @property
    def providers(self) -> tuple[DiagnosticScopeDiscoveryProvider, ...]:
        return self._providers

    def resolve_for_reference(
        self,
        reference: DiagnosticScopeReference,
    ) -> DiagnosticScopeDiscoveryProvider | None:
        reference_kind = reference.kind
        for provider in self._providers:
            if provider.supported_reference_kind is reference_kind:
                return provider
        return None
