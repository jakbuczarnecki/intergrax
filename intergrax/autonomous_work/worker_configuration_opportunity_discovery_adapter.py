# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production join adapter — catalog CapabilityIdentityKey + Integrations opportunity."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityDiscoveryDisposition,
    CapabilityOperationCoverage,
    WorkerAutonomyLevel,
    WorkerCapabilityCandidate,
    WorkerCapabilityCandidateKind,
    WorkerCapabilityDiscoveryLayerOutcome,
    WorkerCapabilityDiscoveryRequest,
    derive_worker_capability_candidate_id,
)
from intergrax.contracts.autonomous_work.references import ProblemReference
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.integrations.contracts.existing_capability_configuration import (
    ExistingCapabilityConfigurationCatalogCorrelationKey,
)
from intergrax.integrations.contracts.existing_capability_configuration_opportunity import (
    ExistingCapabilityConfigurationOpportunity,
    derive_configuration_catalog_correlation_key_from_opportunity,
)


@runtime_checkable
class WorkerConfigurationDiscoveryTenantScopePort(Protocol):
    """Host-resolved tenant for configuration discovery — must match opportunity store scope."""

    def tenant_id_for_discovery(
        self,
        request: WorkerCapabilityDiscoveryRequest,
    ) -> str: ...


@runtime_checkable
class WorkerConfigurationCatalogIdentityVisibilityPort(Protocol):
    """Scope-visible catalog tool identities — Capability Catalog owns identity truth."""

    def scope_visible_tool_identities(
        self,
        request: WorkerCapabilityDiscoveryRequest,
        *,
        tenant_id: str,
    ) -> tuple[CapabilityIdentityKey, ...]: ...


@runtime_checkable
class WorkerConfigurationOpportunityCatalogCorrelationPort(Protocol):
    """Host-composed typed join from catalog identity to integration correlation key."""

    def correlation_key_for_capability_identity(
        self,
        identity: CapabilityIdentityKey,
    ) -> ExistingCapabilityConfigurationCatalogCorrelationKey | None: ...


@runtime_checkable
class TenantScopedConfigurationOpportunityListingPort(Protocol):
    """Integrations-owned tenant opportunity listing — no realization or ranking."""

    def list_opportunities(
        self,
        *,
        tenant_id: str,
    ) -> tuple[ExistingCapabilityConfigurationOpportunity, ...]: ...


def _capability_ref_for_identity(key: CapabilityIdentityKey) -> str:
    return f"{key.kind.value}:{key.source_kind.value}:{key.source_id}:{key.logical_id}"


def _evidence_for_identity(key: CapabilityIdentityKey) -> tuple[ProblemReference, ...]:
    return (
        ProblemReference(
            "capability/catalog/"
            f"{key.kind.value}/"
            f"{key.source_kind.value}/"
            f"{key.source_id}/"
            f"{key.logical_id}"
        ),
    )


class WorkerConfigurationOpportunityDiscoveryAdapter:
    """Join/projection only — no provider selection, realization, or CONFIGURE_EXISTING decision."""

    def __init__(
        self,
        *,
        tenant_scope: WorkerConfigurationDiscoveryTenantScopePort,
        catalog_visibility: WorkerConfigurationCatalogIdentityVisibilityPort,
        catalog_correlation: WorkerConfigurationOpportunityCatalogCorrelationPort,
        opportunity_listing: TenantScopedConfigurationOpportunityListingPort,
    ) -> None:
        self._tenant_scope = tenant_scope
        self._catalog_visibility = catalog_visibility
        self._catalog_correlation = catalog_correlation
        self._opportunity_listing = opportunity_listing

    def discover(
        self,
        request: WorkerCapabilityDiscoveryRequest,
    ) -> WorkerCapabilityDiscoveryLayerOutcome:
        tenant_id = self._tenant_scope.tenant_id_for_discovery(request)
        visible = self._catalog_visibility.scope_visible_tool_identities(
            request,
            tenant_id=tenant_id,
        )
        opportunities = self._opportunity_listing.list_opportunities(tenant_id=tenant_id)

        identity_by_key: dict[
            ExistingCapabilityConfigurationCatalogCorrelationKey,
            CapabilityIdentityKey,
        ] = {}
        for identity in visible:
            if identity.kind is not CapabilityKind.TOOL:
                continue
            correlation = self._catalog_correlation.correlation_key_for_capability_identity(
                identity,
            )
            if correlation is None:
                continue
            existing = identity_by_key.get(correlation)
            if existing is not None and existing.sort_key != identity.sort_key:
                return WorkerCapabilityDiscoveryLayerOutcome(
                    disposition=CapabilityDiscoveryDisposition.CONFLICT,
                )
            identity_by_key[correlation] = identity

        opportunity_by_key: dict[
            ExistingCapabilityConfigurationCatalogCorrelationKey,
            ExistingCapabilityConfigurationOpportunity,
        ] = {}
        for opportunity in opportunities:
            if opportunity.tenant_id != tenant_id:
                return WorkerCapabilityDiscoveryLayerOutcome(
                    disposition=CapabilityDiscoveryDisposition.CONFLICT,
                )
            key = derive_configuration_catalog_correlation_key_from_opportunity(opportunity)
            existing = opportunity_by_key.get(key)
            if existing is not None and existing.configuration_ref != opportunity.configuration_ref:
                return WorkerCapabilityDiscoveryLayerOutcome(
                    disposition=CapabilityDiscoveryDisposition.CONFLICT,
                )
            opportunity_by_key[key] = opportunity

        required_ops = request.need.required_operations
        candidates: list[WorkerCapabilityCandidate] = []
        for key, identity in identity_by_key.items():
            opportunity = opportunity_by_key.get(key)
            if opportunity is None:
                continue
            candidates.append(
                WorkerCapabilityCandidate(
                    candidate_id=derive_worker_capability_candidate_id(
                        candidate_kind=WorkerCapabilityCandidateKind.EXISTING_CONFIGURATION,
                        capability_ref=_capability_ref_for_identity(identity),
                        configuration_ref=str(opportunity.configuration_ref),
                    ),
                    candidate_kind=WorkerCapabilityCandidateKind.EXISTING_CONFIGURATION,
                    capability_ref=_capability_ref_for_identity(identity),
                    source_domain=identity.source_id,
                    operations=required_ops,
                    risk_class=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
                    evidence_refs=_evidence_for_identity(identity),
                    discovered_at=request.need.requested_at,
                    operation_coverage=CapabilityOperationCoverage.EXACT,
                    configuration_ref=str(opportunity.configuration_ref),
                    capability_identity=identity,
                ),
            )

        if not candidates:
            return WorkerCapabilityDiscoveryLayerOutcome(
                disposition=CapabilityDiscoveryDisposition.NO_MATCH,
            )
        return WorkerCapabilityDiscoveryLayerOutcome(
            disposition=CapabilityDiscoveryDisposition.MATCH_FOUND,
            candidates=tuple(sorted(candidates, key=lambda item: item.candidate_id)),
        )


__all__ = [
    "TenantScopedConfigurationOpportunityListingPort",
    "WorkerConfigurationCatalogIdentityVisibilityPort",
    "WorkerConfigurationDiscoveryTenantScopePort",
    "WorkerConfigurationOpportunityCatalogCorrelationPort",
    "WorkerConfigurationOpportunityDiscoveryAdapter",
]
