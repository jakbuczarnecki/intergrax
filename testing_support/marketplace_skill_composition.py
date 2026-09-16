# © Artur Czarnecki. All rights reserved.

"""ME-15 reference composition: Marketplace → Skill lifecycle → composition/binding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from intergrax.capability_catalog import (
    AvailabilityPreservingGovernanceEvaluator,
    FederatedCapabilityCatalog,
)
from intergrax.contracts.capability_catalog import (
    CapabilityDiscoveryQuery,
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
    CapabilityGovernanceContext,
    CapabilityGovernancePosture,
    CapabilityKind,
    CapabilityReleaseIdentity,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.marketplace.handoff_traceability import (
    CapabilityHandoffConsumerError,
    CapabilityHandoffEnvelope,
    CapabilityHandoffConsumerTarget,
)
from intergrax.contracts.marketplace.lifecycle_handoff_intent import (
    MarketplaceLifecycleHandoffIntent,
)
from intergrax.contracts.marketplace.lifecycle_handoff_outcome import (
    MarketplaceLifecycleHandoffStatus,
)
from intergrax.contracts.marketplace.lifecycle_handoff_request import (
    MarketplaceLifecycleDomainPayload,
)
from intergrax.contracts.marketplace.listing_record import MarketplaceListingRecord
from intergrax.contracts.marketplace.query_context import MarketplaceQueryContext
from intergrax.contracts.marketplace.visibility import (
    MarketplaceVisibility,
    MarketplaceVisibilityScope,
)
from intergrax.contracts.skills.marketplace_lifecycle_handoff import (
    SkillLifecycleHandoffPayload,
)
from intergrax.marketplace import (
    MarketplaceCapabilityCatalogSource,
    MarketplaceCatalogService,
    MarketplaceDiscoveryService,
)
from intergrax.marketplace.handoff import (
    LifecycleHandoffResolver,
    MarketplaceLifecycleHandoffService,
    SkillMarketplaceLifecycleHandoffHandler,
)
from intergrax.marketplace.handoff.adapters.skill_acquisition_bridge import (
    SkillMarketplaceAcquisitionBridge,
)
from intergrax.marketplace.handoff_traceability import (
    CapabilityHandoffDeliveryService,
    InMemoryCapabilityHandoffDeliveryAdmission,
    InMemoryCapabilityHandoffTraceEvidenceConsumer,
    MarketplaceDiscoveryHandoffOrchestrator,
)
from intergrax.marketplace.handoff_traceability.lifecycle_bridge import (
    lifecycle_handoff_request_from_envelope,
)
from intergrax.skills.catalog import SkillCatalogProviderRegistry
from intergrax.skills.dynamic_acquisition import (
    DynamicSkillAcquisitionRequest,
    DynamicSkillAcquisitionService,
)
from intergrax.skills.execution_binding import resolve_skill_composition_from_profile
from intergrax.skills.host_lifecycle import SkillHostLifecycleService
from intergrax.skills.identity import SkillDiscoveryCandidateIdentity, SkillPackageCandidate
from testing_support.canonical_me15_reference_skill import (
    ME15_DIGEST_V1,
    ME15_DIGEST_V2,
    ME15_PACKAGE_REFERENCE_V1,
    ME15_SKILL_LOGICAL_ID,
    ME15_VERSION_V1,
    ME15_VERSION_V2,
    instruction_marker_for_release,
)
from testing_support.me15_skill_binding_materializer import Me15SkillHostBindingMaterializer
from testing_support.me15_skill_catalog_provider import (
    ME15_CATALOG_ENTRY_ID,
    ME15_CATALOG_SOURCE_ID,
    Me15SkillCatalogProvider,
)

ME15_HOST_PROFILE_ID: Final = "host-profile-me15"

ME15_CAPABILITY_SOURCE: Final = CapabilitySourceIdentity(
    source_id="official.intergrax.me15",
    source_kind=CapabilitySourceKind.OFFICIAL,
)
ME15_LISTING_ID: Final = "listing-me15-canonical-instruction"


@dataclass(frozen=True, slots=True)
class MarketplaceSkillE2EProofEvidence:
    discovery_correlation_id: str
    selection_id: str
    handoff_id: str
    selected_release: CapabilityReleaseIdentity
    lifecycle_request_id: str
    skill_operation_id: str
    lifecycle_domain_reference: str
    host_profile_id: str
    resolved_version_label: str
    resolved_content_digest: str
    registry_skill_id: str
    bound_version_label: str
    bound_content_digest: str
    snapshot_digest: str
    profile_skill_ids: tuple[str, ...]


def me15_default_listing_v1() -> MarketplaceListingRecord:
    return _marketplace_listing_record(
        version_label=ME15_VERSION_V1,
        content_digest=ME15_DIGEST_V1,
    )


def me15_listing_v2() -> MarketplaceListingRecord:
    return _marketplace_listing_record(
        version_label=ME15_VERSION_V2,
        content_digest=ME15_DIGEST_V2,
    )


def _marketplace_listing_record(
    *,
    version_label: str,
    content_digest: str,
    tenant_id: str | None = None,
    organization_id: str | None = None,
) -> MarketplaceListingRecord:
    visibility = None
    if organization_id is not None:
        visibility = MarketplaceVisibility(
            scope=MarketplaceVisibilityScope.ORGANIZATION_PRIVATE,
            organization_id=organization_id,
        )
    elif tenant_id is not None:
        visibility = MarketplaceVisibility(
            scope=MarketplaceVisibilityScope.TENANT_PRIVATE,
            tenant_id=tenant_id,
        )
    return MarketplaceListingRecord(
        kind=CapabilityKind.SKILL,
        logical_id=ME15_SKILL_LOGICAL_ID,
        listing_id=ME15_LISTING_ID,
        display_label="ME-15 Canonical Instruction Skill",
        publisher="publisher:me15",
        version_label=version_label,
        content_digest=content_digest,
        package_reference=ME15_PACKAGE_REFERENCE_V1,
        visibility=visibility,
    )


def build_marketplace_catalog_service(
    *records: MarketplaceListingRecord,
) -> MarketplaceCatalogService:
    source = MarketplaceCapabilityCatalogSource(
        source=ME15_CAPABILITY_SOURCE,
        records=records,
    )
    catalog = FederatedCapabilityCatalog((source,))
    return MarketplaceCatalogService(catalog=catalog, marketplace_sources=(source,))


def skill_discovery_identity_from_release(
    *,
    release: CapabilityReleaseIdentity,
) -> SkillDiscoveryCandidateIdentity:
    discovery = release.discovery
    if release.version_label is None or release.content_digest is None:
        raise ValueError("release must include version_label and content_digest")
    return SkillDiscoveryCandidateIdentity(
        catalog_source_id=ME15_CATALOG_SOURCE_ID,
        package=SkillPackageCandidate(
            logical_skill_id=discovery.logical.logical_id,
            package_reference=release.package_reference or ME15_PACKAGE_REFERENCE_V1,
            package_version=release.version_label,
            package_digest=release.content_digest,
        ),
    )


def build_dynamic_skill_acquisition_request(
    *,
    payload: SkillLifecycleHandoffPayload,
    envelope: CapabilityHandoffEnvelope,
) -> DynamicSkillAcquisitionRequest:
    identity_key = CapabilityIdentityKey.from_discovery_identity(
        envelope.selected_release.discovery,
    )
    if payload.capability_identity_key != identity_key:
        raise ValueError("handoff payload capability_identity_key must match envelope release")
    return DynamicSkillAcquisitionRequest(
        operation_id=payload.operation_id,
        host_profile_id=payload.host_profile_id,
        capability_identity_key=identity_key,
        selected_identity=skill_discovery_identity_from_release(
            release=envelope.selected_release,
        ),
        catalog_entry_id=ME15_CATALOG_ENTRY_ID,
    )


@dataclass
class MarketplaceSkillHandoffConsumer:
    """ME-15 consumer: ME-RB4 Skill handler + production Skill acquisition bridge."""

    acquisition: DynamicSkillAcquisitionService
    host_profile_id: str
    _delivered_handoffs: list[str] = field(default_factory=list)
    last_envelope: CapabilityHandoffEnvelope | None = field(default=None, init=False)
    last_lifecycle_domain_reference: str | None = field(default=None, init=False)

    @property
    def consumer_id(self) -> str:
        return "me15.skill_domain.handoff_consumer"

    def consume(self, envelope: CapabilityHandoffEnvelope) -> None:
        identity_key = CapabilityIdentityKey.from_discovery_identity(
            envelope.selected_release.discovery,
        )
        operation_id = envelope.handoff_id
        payload = SkillLifecycleHandoffPayload(
            operation_id=operation_id,
            host_profile_id=self.host_profile_id,
            capability_identity_key=identity_key,
        )
        lifecycle_request = lifecycle_handoff_request_from_envelope(
            envelope,
            request_id=f"lifecycle:{envelope.handoff_id}",
            intent=MarketplaceLifecycleHandoffIntent.REQUEST_LIFECYCLE,
            domain_payload=MarketplaceLifecycleDomainPayload(skill=payload),
            correlation_id=envelope.discovery_correlation_id,
        )

        def request_factory(
            handoff_payload: SkillLifecycleHandoffPayload,
        ) -> DynamicSkillAcquisitionRequest:
            return build_dynamic_skill_acquisition_request(
                payload=handoff_payload,
                envelope=envelope,
            )

        bridge = SkillMarketplaceAcquisitionBridge(
            self.acquisition,
            request_factory,
        )
        handler = SkillMarketplaceLifecycleHandoffHandler(bridge)
        resolver = LifecycleHandoffResolver({CapabilityKind.SKILL: handler})
        outcome = MarketplaceLifecycleHandoffService(resolver).handoff(lifecycle_request)
        if outcome.status is not MarketplaceLifecycleHandoffStatus.ACCEPTED:
            raise CapabilityHandoffConsumerError(
                f"skill domain rejected marketplace handoff: {outcome.reason_detail}",
            )
        self.last_envelope = envelope
        self.last_lifecycle_domain_reference = outcome.domain_reference
        self._delivered_handoffs.append(envelope.handoff_id)


@dataclass(frozen=True, slots=True)
class MarketplaceSkillCompositionProofStack:
    lifecycle: SkillHostLifecycleService
    acquisition: DynamicSkillAcquisitionService
    catalog_provider: Me15SkillCatalogProvider
    catalog_service: MarketplaceCatalogService
    orchestrator: MarketplaceDiscoveryHandoffOrchestrator
    handoff_consumer: MarketplaceSkillHandoffConsumer
    delivery_admission: InMemoryCapabilityHandoffDeliveryAdmission
    delivery_service: CapabilityHandoffDeliveryService

    @classmethod
    def build(
        cls,
        *,
        listing_records: tuple[MarketplaceListingRecord, ...] | None = None,
        lifecycle: SkillHostLifecycleService | None = None,
        catalog_provider: Me15SkillCatalogProvider | None = None,
    ) -> MarketplaceSkillCompositionProofStack:
        resolved_provider = catalog_provider or Me15SkillCatalogProvider()
        resolved_lifecycle = lifecycle or SkillHostLifecycleService(
            host_profile_id=ME15_HOST_PROFILE_ID,
        )
        materializer = Me15SkillHostBindingMaterializer(
            resolved_lifecycle.registry,
            catalog_source_id=resolved_provider.catalog_source_id,
        )
        acquisition = DynamicSkillAcquisitionService(
            catalog_registry=SkillCatalogProviderRegistry(
                {resolved_provider.catalog_source_id: resolved_provider},
            ),
            binding=resolved_lifecycle,
            materializer=materializer,
        )
        records = listing_records or (me15_default_listing_v1(),)
        catalog_service = build_marketplace_catalog_service(*records)
        handoff_consumer = MarketplaceSkillHandoffConsumer(
            acquisition=acquisition,
            host_profile_id=resolved_lifecycle.host_profile_id,
        )
        delivery_admission = InMemoryCapabilityHandoffDeliveryAdmission()
        delivery = CapabilityHandoffDeliveryService(
            consumer=handoff_consumer,
            delivery_admission=delivery_admission,
            trace_evidence_consumer=InMemoryCapabilityHandoffTraceEvidenceConsumer(),
        )
        orchestrator = MarketplaceDiscoveryHandoffOrchestrator(
            catalog_service=catalog_service,
            discovery_service=MarketplaceDiscoveryService.with_defaults(),
            governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
            governance_context=CapabilityGovernanceContext(
                posture=CapabilityGovernancePosture.STRICT,
            ),
            delivery_service=delivery,
        )
        return cls(
            lifecycle=resolved_lifecycle,
            acquisition=acquisition,
            catalog_provider=resolved_provider,
            catalog_service=catalog_service,
            orchestrator=orchestrator,
            handoff_consumer=handoff_consumer,
            delivery_admission=delivery_admission,
            delivery_service=delivery,
        )

    def marketplace_identity_key(self) -> CapabilityIdentityKey:
        snapshot = self.catalog_service._catalog.snapshot()
        entry = next(
            item
            for item in snapshot.entries
            if item.identity.logical.logical_id == ME15_SKILL_LOGICAL_ID
        )
        return CapabilityIdentityKey.from_discovery_identity(entry.identity)

    def run_marketplace_skill_e2e(
        self,
        *,
        tenant_id: str | None = None,
        discovery_correlation_id: str = "discovery-corr-me15",
        selection_id: str = "selection-me15",
        handoff_id: str = "handoff-me15",
    ) -> MarketplaceSkillE2EProofEvidence:
        assert not self.lifecycle.is_bound(ME15_SKILL_LOGICAL_ID)
        self.orchestrator.execute_explicit_selection_handoff(
            discovery_query=_discovery_query(),
            marketplace_query_context=MarketplaceQueryContext(tenant_id=tenant_id),
            selected_identity_key=self.marketplace_identity_key(),
            consumer_target=CapabilityHandoffConsumerTarget.SKILL_DOMAIN,
            selector_id="operator.me15.explicit",
            discovery_correlation_id=discovery_correlation_id,
            selection_id=selection_id,
            handoff_id=handoff_id,
        )
        envelope = self.handoff_consumer.last_envelope
        assert envelope is not None
        selected_release = envelope.selected_release
        assert self.lifecycle.is_bound(ME15_SKILL_LOGICAL_ID)
        binding = self.lifecycle.binding_metadata(ME15_SKILL_LOGICAL_ID)
        assert binding is not None
        composition = resolve_skill_composition_from_profile(
            self.lifecycle.skill_profile,
            skill_registry=self.lifecycle.registry,
        )
        marker = instruction_marker_for_release(selected_release)
        assert marker in composition.pack.prompt_instruction_ids
        assert binding.version_label == selected_release.version_label
        assert binding.content_digest == selected_release.content_digest
        registered = self.lifecycle.registry.get(ME15_SKILL_LOGICAL_ID)
        assert registered.manifest.version == selected_release.version_label
        enabled = tuple(
            skill_id
            for skill_id in self.lifecycle.registry.skill_ids()
            if self.lifecycle.skill_profile.is_skill_enabled(skill_id)
        )
        return MarketplaceSkillE2EProofEvidence(
            discovery_correlation_id=discovery_correlation_id,
            selection_id=selection_id,
            handoff_id=handoff_id,
            selected_release=selected_release,
            lifecycle_request_id=f"lifecycle:{handoff_id}",
            skill_operation_id=handoff_id,
            lifecycle_domain_reference=self.handoff_consumer.last_lifecycle_domain_reference or "",
            host_profile_id=self.lifecycle.host_profile_id,
            resolved_version_label=binding.version_label,
            resolved_content_digest=binding.content_digest,
            registry_skill_id=ME15_SKILL_LOGICAL_ID,
            bound_version_label=binding.version_label,
            bound_content_digest=binding.content_digest,
            snapshot_digest=composition.pack.snapshot_digest,
            profile_skill_ids=enabled,
        )


def _discovery_query() -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
    )


__all__ = [
    "ME15_CAPABILITY_SOURCE",
    "ME15_HOST_PROFILE_ID",
    "ME15_LISTING_ID",
    "MarketplaceSkillCompositionProofStack",
    "MarketplaceSkillE2EProofEvidence",
    "MarketplaceSkillHandoffConsumer",
    "_marketplace_listing_record",
    "build_dynamic_skill_acquisition_request",
    "build_marketplace_catalog_service",
    "me15_default_listing_v1",
    "me15_listing_v2",
    "skill_discovery_identity_from_release",
]
