# © Artur Czarnecki. All rights reserved.

"""ME-14 reference composition: Marketplace → Tool lifecycle → catalog tool execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
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
from intergrax.contracts.tools.marketplace_lifecycle_handoff import (
    ToolLifecycleHandoffPayload,
)
from intergrax.marketplace import (
    MarketplaceCapabilityCatalogSource,
    MarketplaceCatalogService,
    MarketplaceDiscoveryService,
)
from intergrax.marketplace.handoff import (
    LifecycleHandoffResolver,
    MarketplaceLifecycleHandoffService,
    ToolMarketplaceLifecycleHandoffHandler,
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
from testing_support.canonical_me14_echo_tool import (
    ME14_DIGEST_V1,
    ME14_DIGEST_V2,
    ME14_PACKAGE_REFERENCE_V1,
    ME14_TOOL_LOGICAL_ID,
    ME14_VERSION_V1,
    ME14_VERSION_V2,
    expected_output_for_release,
)
from intergrax.marketplace.handoff.adapters.tool_acquisition_bridge import (
    ToolMarketplaceAcquisitionBridge,
)
from intergrax.tools.catalog import ToolCatalogProviderRegistry
from intergrax.tools.dynamic_acquisition import (
    DynamicToolAcquisitionRequest,
    DynamicToolAcquisitionService,
)
from intergrax.tools.host_lifecycle import ToolHostLifecycleService
from intergrax.tools.identity import ToolDiscoveryCandidateIdentity, ToolPackageCandidate
from testing_support.me14_tool_activation_materializer import Me14ToolHostActivationMaterializer
from testing_support.me14_tool_catalog_provider import (
    ME14_CATALOG_ENTRY_ID,
    ME14_CATALOG_SOURCE_ID,
    Me14ToolCatalogProvider,
)
from testing_support.me14_tool_harness_execution import execute_me14_tool_via_host_execution_engine
from testing_support.reference_tool_host_lifecycle_service import ReferenceToolHostLifecycleService

ME14_HOST_PROFILE_ID: Final = "host-profile-me14"

ME14_CAPABILITY_SOURCE: Final = CapabilitySourceIdentity(
    source_id="official.intergrax.me14",
    source_kind=CapabilitySourceKind.OFFICIAL,
)
ME14_LISTING_ID: Final = "listing-me14-canonical-echo"


@dataclass(frozen=True, slots=True)
class MarketplaceToolE2EProofEvidence:
    discovery_correlation_id: str
    selection_id: str
    handoff_id: str
    selected_release: CapabilityReleaseIdentity
    lifecycle_request_id: str
    acquisition_operation_id: str
    lifecycle_domain_reference: str
    host_profile_id: str
    resolved_version_label: str
    resolved_content_digest: str
    registry_tool_id: str
    activated_version_label: str
    activated_content_digest: str
    execution_tool_id: str
    execution_task_id: str | None
    execution_result: str


def me14_default_listing_v1() -> MarketplaceListingRecord:
    return _marketplace_listing_record(
        version_label=ME14_VERSION_V1,
        content_digest=ME14_DIGEST_V1,
    )


def me14_listing_v2() -> MarketplaceListingRecord:
    return _marketplace_listing_record(
        version_label=ME14_VERSION_V2,
        content_digest=ME14_DIGEST_V2,
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
        kind=CapabilityKind.TOOL,
        logical_id=ME14_TOOL_LOGICAL_ID,
        listing_id=ME14_LISTING_ID,
        display_label="ME-14 Canonical Echo Tool",
        publisher="publisher:me14",
        version_label=version_label,
        content_digest=content_digest,
        package_reference=ME14_PACKAGE_REFERENCE_V1,
        visibility=visibility,
    )


def build_marketplace_catalog_service(
    *records: MarketplaceListingRecord,
) -> MarketplaceCatalogService:
    source = MarketplaceCapabilityCatalogSource(
        source=ME14_CAPABILITY_SOURCE,
        records=records,
    )
    catalog = FederatedCapabilityCatalog((source,))
    return MarketplaceCatalogService(catalog=catalog, marketplace_sources=(source,))


def tool_discovery_identity_from_release(
    *,
    release: CapabilityReleaseIdentity,
) -> ToolDiscoveryCandidateIdentity:
    discovery = release.discovery
    if release.version_label is None or release.content_digest is None:
        raise ValueError("release must include version_label and content_digest")
    return ToolDiscoveryCandidateIdentity(
        catalog_source_id=ME14_CATALOG_SOURCE_ID,
        package=ToolPackageCandidate(
            logical_tool_id=discovery.logical.logical_id,
            package_reference=release.package_reference or ME14_PACKAGE_REFERENCE_V1,
            package_version=release.version_label,
            package_digest=release.content_digest,
        ),
    )


def build_dynamic_tool_acquisition_request(
    *,
    payload: ToolLifecycleHandoffPayload,
    envelope: CapabilityHandoffEnvelope,
) -> DynamicToolAcquisitionRequest:
    identity_key = CapabilityIdentityKey.from_discovery_identity(
        envelope.selected_release.discovery,
    )
    if payload.capability_identity_key != identity_key:
        raise ValueError("handoff payload capability_identity_key must match envelope release")
    return DynamicToolAcquisitionRequest(
        operation_id=payload.operation_id,
        host_profile_id=payload.host_profile_id,
        capability_identity_key=identity_key,
        selected_identity=tool_discovery_identity_from_release(
            release=envelope.selected_release,
        ),
        catalog_entry_id=ME14_CATALOG_ENTRY_ID,
    )


@dataclass
class MarketplaceToolHandoffConsumer:
    """ME-10 consumer: ME-RB4 Tool handler + production Tool acquisition bridge."""

    acquisition: DynamicToolAcquisitionService
    host_profile_id: str
    _delivered_handoffs: list[str] = field(default_factory=list)
    last_envelope: CapabilityHandoffEnvelope | None = field(default=None, init=False)
    last_lifecycle_domain_reference: str | None = field(default=None, init=False)

    @property
    def consumer_id(self) -> str:
        return "me14.tool_domain.handoff_consumer"

    def consume(self, envelope: CapabilityHandoffEnvelope) -> None:
        identity_key = CapabilityIdentityKey.from_discovery_identity(
            envelope.selected_release.discovery,
        )
        operation_id = envelope.handoff_id
        payload = ToolLifecycleHandoffPayload(
            operation_id=operation_id,
            host_profile_id=self.host_profile_id,
            capability_identity_key=identity_key,
        )
        lifecycle_request = lifecycle_handoff_request_from_envelope(
            envelope,
            request_id=f"lifecycle:{envelope.handoff_id}",
            intent=MarketplaceLifecycleHandoffIntent.REQUEST_LIFECYCLE,
            domain_payload=MarketplaceLifecycleDomainPayload(tool=payload),
            correlation_id=envelope.discovery_correlation_id,
        )

        def request_factory(
            handoff_payload: ToolLifecycleHandoffPayload,
        ) -> DynamicToolAcquisitionRequest:
            return build_dynamic_tool_acquisition_request(
                payload=handoff_payload,
                envelope=envelope,
            )

        bridge = ToolMarketplaceAcquisitionBridge(
            self.acquisition,
            request_factory,
        )
        handler = ToolMarketplaceLifecycleHandoffHandler(bridge)
        resolver = LifecycleHandoffResolver({CapabilityKind.TOOL: handler})
        outcome = MarketplaceLifecycleHandoffService(resolver).handoff(lifecycle_request)
        if outcome.status is not MarketplaceLifecycleHandoffStatus.ACCEPTED:
            raise CapabilityHandoffConsumerError(
                f"tool domain rejected marketplace handoff: {outcome.reason_detail}",
            )
        self.last_envelope = envelope
        self.last_lifecycle_domain_reference = outcome.domain_reference
        self._delivered_handoffs.append(envelope.handoff_id)


@dataclass(frozen=True, slots=True)
class MarketplaceToolExecutionProofStack:
    lifecycle: ToolHostLifecycleService
    acquisition: DynamicToolAcquisitionService
    catalog_provider: Me14ToolCatalogProvider
    catalog_service: MarketplaceCatalogService
    orchestrator: MarketplaceDiscoveryHandoffOrchestrator
    handoff_consumer: MarketplaceToolHandoffConsumer
    delivery_admission: InMemoryCapabilityHandoffDeliveryAdmission
    delivery_service: CapabilityHandoffDeliveryService
    execution_tmp_root: Path | None = None

    @classmethod
    def build(
        cls,
        *,
        listing_records: tuple[MarketplaceListingRecord, ...] | None = None,
        lifecycle: ToolHostLifecycleService | ReferenceToolHostLifecycleService | None = None,
        catalog_provider: Me14ToolCatalogProvider | None = None,
        execution_tmp_root: Path | None = None,
    ) -> MarketplaceToolExecutionProofStack:
        if isinstance(lifecycle, ReferenceToolHostLifecycleService):
            raise TypeError(
                "ReferenceToolHostLifecycleService is not canonical ME-14-C1 lifecycle authority",
            )
        resolved_provider = catalog_provider or Me14ToolCatalogProvider()
        resolved_lifecycle = lifecycle or ToolHostLifecycleService(
            host_profile_id=ME14_HOST_PROFILE_ID,
        )
        materializer = Me14ToolHostActivationMaterializer(
            resolved_lifecycle.registry,
            catalog_source_id=resolved_provider.catalog_source_id,
        )
        acquisition = DynamicToolAcquisitionService(
            catalog_registry=ToolCatalogProviderRegistry(
                {resolved_provider.catalog_source_id: resolved_provider},
            ),
            activation=resolved_lifecycle,
            materializer=materializer,
        )
        records = listing_records or (me14_default_listing_v1(),)
        catalog_service = build_marketplace_catalog_service(*records)
        handoff_consumer = MarketplaceToolHandoffConsumer(
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
            execution_tmp_root=execution_tmp_root,
        )

    async def execute_tool_via_host_execution_engine(
        self,
        tmp_path: Path,
    ) -> tuple[str, str, str | None]:
        registry = self.lifecycle.registry_read()
        if not registry.has(ME14_TOOL_LOGICAL_ID):
            raise RuntimeError("tool not active in domain registry")
        return await execute_me14_tool_via_host_execution_engine(
            registry=registry,
            tool_logical_id=ME14_TOOL_LOGICAL_ID,
            tmp_path=tmp_path,
        )

    def run_marketplace_tool_e2e(
        self,
        *,
        tenant_id: str | None = None,
        discovery_correlation_id: str = "discovery-corr-me14",
        selection_id: str = "selection-me14",
        handoff_id: str = "handoff-me14",
        execution_tmp_path: Path,
    ) -> MarketplaceToolE2EProofEvidence:
        assert not self.lifecycle.is_active(ME14_TOOL_LOGICAL_ID)
        self.orchestrator.execute_explicit_selection_handoff(
            discovery_query=_discovery_query(),
            marketplace_query_context=MarketplaceQueryContext(tenant_id=tenant_id),
            selected_identity_key=self.marketplace_identity_key(),
            consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
            selector_id="operator.me14.explicit",
            discovery_correlation_id=discovery_correlation_id,
            selection_id=selection_id,
            handoff_id=handoff_id,
        )
        envelope = self.handoff_consumer.last_envelope
        assert envelope is not None
        selected_release = envelope.selected_release
        assert self.lifecycle.is_active(ME14_TOOL_LOGICAL_ID)
        activation = self.lifecycle.activation_metadata(ME14_TOOL_LOGICAL_ID)
        assert activation is not None
        execution_tool_id, execution_result, execution_task_id = asyncio.run(
            self.execute_tool_via_host_execution_engine(execution_tmp_path),
        )
        expected = expected_output_for_release(selected_release)
        assert execution_result == expected
        assert activation.version_label == selected_release.version_label
        assert activation.content_digest == selected_release.content_digest
        return MarketplaceToolE2EProofEvidence(
            discovery_correlation_id=discovery_correlation_id,
            selection_id=selection_id,
            handoff_id=handoff_id,
            selected_release=selected_release,
            lifecycle_request_id=f"lifecycle:{handoff_id}",
            acquisition_operation_id=handoff_id,
            lifecycle_domain_reference=self.handoff_consumer.last_lifecycle_domain_reference or "",
            host_profile_id=self.lifecycle.host_profile_id,
            resolved_version_label=activation.version_label,
            resolved_content_digest=activation.content_digest,
            registry_tool_id=ME14_TOOL_LOGICAL_ID,
            activated_version_label=activation.version_label,
            activated_content_digest=activation.content_digest,
            execution_tool_id=execution_tool_id,
            execution_task_id=execution_task_id,
            execution_result=execution_result,
        )


    def marketplace_identity_key(self) -> CapabilityIdentityKey:
        snapshot = self.catalog_service._catalog.snapshot()
        entry = next(
            item
            for item in snapshot.entries
            if item.identity.logical.logical_id == ME14_TOOL_LOGICAL_ID
        )
        return CapabilityIdentityKey.from_discovery_identity(entry.identity)


def _discovery_query() -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
    )


__all__ = [
    "ME14_CAPABILITY_SOURCE",
    "ME14_LISTING_ID",
    "MarketplaceToolE2EProofEvidence",
    "MarketplaceToolExecutionProofStack",
    "MarketplaceToolHandoffConsumer",
    "_marketplace_listing_record",
    "build_dynamic_tool_acquisition_request",
    "build_marketplace_catalog_service",
    "me14_default_listing_v1",
    "me14_listing_v2",
    "tool_discovery_identity_from_release",
]
