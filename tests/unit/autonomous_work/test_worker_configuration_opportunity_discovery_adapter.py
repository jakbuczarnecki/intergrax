# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from intergrax.autonomous_work.worker_configuration_opportunity_discovery_adapter import (
    TenantScopedConfigurationOpportunityListingPort,
    WorkerConfigurationCatalogIdentityVisibilityPort,
    WorkerConfigurationDiscoveryTenantScopePort,
    WorkerConfigurationOpportunityCatalogCorrelationPort,
    WorkerConfigurationOpportunityDiscoveryAdapter,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityDiscoveryDisposition,
    CapabilityNeedKind,
    WorkerCapabilityCandidateKind,
    WorkerCapabilityDiscoveryRequest,
    WorkerCapabilityNeed,
)
from intergrax.contracts.autonomous_work.ids import mint_worker_instance_id
from intergrax.contracts.autonomous_work.profile_reference import (
    CapabilityProfileRef,
    initial_profile_version,
)
from intergrax.contracts.capability_catalog import CapabilityKind, CapabilitySourceKind
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRisk
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.existing_capability_configuration import (
    ExistingCapabilityConfigurationCatalogCorrelationKey,
)
from intergrax.integrations.contracts.existing_capability_configuration_opportunity import (
    ConfigurationOpportunityRef,
    ExistingCapabilityConfigurationOpportunity,
    validate_configuration_opportunity_ref,
)

pytestmark = pytest.mark.unit

_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"
_NOW = datetime(2026, 3, 20, 12, 0, 0, tzinfo=UTC)
_WORKER = mint_worker_instance_id()
_PROFILE = CapabilityProfileRef(profile_id="profile/default", version=initial_profile_version())
_CONFIG_REF = validate_configuration_opportunity_ref("cfg-ref-join-1")
_CORRELATION = ExistingCapabilityConfigurationCatalogCorrelationKey(
    integration_category=IntegrationCategory.RELATIONAL_STORE,
    provider_id="sqlite",
    resource_scope="default",
)


def _identity() -> CapabilityIdentityKey:
    return CapabilityIdentityKey(
        kind=CapabilityKind.TOOL,
        source_id="official.marketplace",
        source_kind=CapabilitySourceKind.OFFICIAL,
        logical_id="tools.database.relational",
    )


def _need() -> WorkerCapabilityNeed:
    return WorkerCapabilityNeed(
        worker_instance_id=_WORKER,
        obstacle_id=f"{_WORKER}:obs-1",
        need_kind=CapabilityNeedKind.TOOL_OPERATION,
        required_operations=("database.query",),
        capability_profile_ref=_PROFILE,
        requested_at=_NOW,
        recovery_decision_id="recovery-1",
    )


def _request() -> WorkerCapabilityDiscoveryRequest:
    return WorkerCapabilityDiscoveryRequest(
        need=_need(),
        profile_ref=_PROFILE,
        worker_instance_id=_WORKER,
    )


def _opportunity(
    *,
    tenant_id: str = _TENANT_A,
    configuration_ref: ConfigurationOpportunityRef = _CONFIG_REF,
) -> ExistingCapabilityConfigurationOpportunity:
    return ExistingCapabilityConfigurationOpportunity(
        configuration_ref=configuration_ref,
        tenant_id=tenant_id,
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id="sqlite",
        resource_scope="default",
        current_revision="rev-1",
        configuration=MagicMock(
            configuration_type="test",
            configuration_version="v1",
            configuration_fingerprint="fp-1",
        ),
        configuration_fingerprint="fp-1",
        risk_classification=ControlPlaneMutationRisk.LOW,
    )


@dataclass
class _StaticTenantScope(WorkerConfigurationDiscoveryTenantScopePort):
    tenant_id: str

    def tenant_id_for_discovery(self, request: WorkerCapabilityDiscoveryRequest) -> str:
        return self.tenant_id


@dataclass
class _StaticVisibility(WorkerConfigurationCatalogIdentityVisibilityPort):
    identities: tuple[CapabilityIdentityKey, ...]

    def scope_visible_tool_identities(
        self,
        request: WorkerCapabilityDiscoveryRequest,
        *,
        tenant_id: str,
    ) -> tuple[CapabilityIdentityKey, ...]:
        return self.identities


@dataclass
class _StaticCorrelation(WorkerConfigurationOpportunityCatalogCorrelationPort):
    key: ExistingCapabilityConfigurationCatalogCorrelationKey | None

    def correlation_key_for_capability_identity(
        self,
        identity: CapabilityIdentityKey,
    ) -> ExistingCapabilityConfigurationCatalogCorrelationKey | None:
        return self.key


@dataclass
class _StaticListing(TenantScopedConfigurationOpportunityListingPort):
    opportunities: tuple[ExistingCapabilityConfigurationOpportunity, ...]

    def list_opportunities(
        self,
        *,
        tenant_id: str,
    ) -> tuple[ExistingCapabilityConfigurationOpportunity, ...]:
        return self.opportunities


def _adapter(
    *,
    tenant_id: str = _TENANT_A,
    identities: tuple[CapabilityIdentityKey, ...] = (_identity(),),
    correlation: ExistingCapabilityConfigurationCatalogCorrelationKey | None = _CORRELATION,
    opportunities: tuple[ExistingCapabilityConfigurationOpportunity, ...] | None = None,
) -> WorkerConfigurationOpportunityDiscoveryAdapter:
    if opportunities is None:
        opportunities = (_opportunity(tenant_id=tenant_id),)
    return WorkerConfigurationOpportunityDiscoveryAdapter(
        tenant_scope=_StaticTenantScope(tenant_id),
        catalog_visibility=_StaticVisibility(identities),
        catalog_correlation=_StaticCorrelation(correlation),
        opportunity_listing=_StaticListing(opportunities),
    )


def test_matching_capability_and_opportunity_emits_existing_configuration_candidate() -> None:
    outcome = _adapter().discover(_request())
    assert outcome.disposition is CapabilityDiscoveryDisposition.MATCH_FOUND
    assert len(outcome.candidates) == 1
    assert outcome.candidates[0].candidate_kind is (
        WorkerCapabilityCandidateKind.EXISTING_CONFIGURATION
    )


def test_candidate_has_typed_capability_identity_key() -> None:
    identity = _identity()
    outcome = _adapter(identities=(identity,)).discover(_request())
    assert outcome.candidates[0].capability_identity == identity


def test_candidate_has_exact_configuration_ref() -> None:
    outcome = _adapter().discover(_request())
    assert outcome.candidates[0].configuration_ref == str(_CONFIG_REF)


def test_no_opportunity_yields_no_candidate() -> None:
    outcome = _adapter(opportunities=()).discover(_request())
    assert outcome.disposition is CapabilityDiscoveryDisposition.NO_MATCH


def test_no_catalog_identity_yields_no_candidate() -> None:
    outcome = _adapter(identities=()).discover(_request())
    assert outcome.disposition is CapabilityDiscoveryDisposition.NO_MATCH


def test_tenant_mismatch_on_opportunity_fails_closed() -> None:
    outcome = _adapter(
        tenant_id=_TENANT_A,
        opportunities=(_opportunity(tenant_id=_TENANT_B),),
    ).discover(_request())
    assert outcome.disposition is CapabilityDiscoveryDisposition.CONFLICT


def test_ambiguous_correlation_fails_closed() -> None:
    other = CapabilityIdentityKey(
        kind=CapabilityKind.TOOL,
        source_id="official.marketplace",
        source_kind=CapabilitySourceKind.OFFICIAL,
        logical_id="tools.database.other",
    )
    outcome = _adapter(identities=(_identity(), other)).discover(_request())
    assert outcome.disposition is CapabilityDiscoveryDisposition.CONFLICT


def test_adapter_does_not_call_realization() -> None:
    listing = MagicMock(spec=TenantScopedConfigurationOpportunityListingPort)
    listing.list_opportunities.return_value = (_opportunity(),)
    adapter = WorkerConfigurationOpportunityDiscoveryAdapter(
        tenant_scope=_StaticTenantScope(_TENANT_A),
        catalog_visibility=_StaticVisibility((_identity(),)),
        catalog_correlation=_StaticCorrelation(_CORRELATION),
        opportunity_listing=listing,
    )
    adapter.discover(_request())
    listing.list_opportunities.assert_called_once_with(tenant_id=_TENANT_A)
    assert not hasattr(listing, "realize")


def test_adapter_does_not_invoke_decision_service() -> None:
    outcome = _adapter().discover(_request())
    assert outcome.candidates
    assert outcome.candidates[0].risk_class.value.startswith("A0")


def test_no_provider_selection_occurs() -> None:
    correlation = MagicMock(spec=WorkerConfigurationOpportunityCatalogCorrelationPort)
    correlation.correlation_key_for_capability_identity.return_value = _CORRELATION
    adapter = WorkerConfigurationOpportunityDiscoveryAdapter(
        tenant_scope=_StaticTenantScope(_TENANT_A),
        catalog_visibility=_StaticVisibility((_identity(),)),
        catalog_correlation=correlation,
        opportunity_listing=_StaticListing((_opportunity(),)),
    )
    adapter.discover(_request())
    correlation.correlation_key_for_capability_identity.assert_called_once()
