# © Artur Czarnecki. All rights reserved.

"""ME-14 reference composition: Marketplace → Tool lifecycle → catalog tool execution."""

from __future__ import annotations

import asyncio
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
from testing_support.reference_tool_host_lifecycle_service import (
    ME14_HOST_PROFILE_ID,
    ReferenceToolHostLifecycleService,
    ToolActivationRequest,
)
from testing_support.tool_marketplace_acquisition_bridge import (
    ToolMarketplaceAcquisitionBridge,
)

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
    lifecycle_domain_reference: str
    host_profile_id: str
    registry_tool_id: str
    activated_version_label: str
    activated_content_digest: str
    execution_tool_id: str
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
) -> MarketplaceListingRecord:
    visibility = None
    if tenant_id is not None:
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


def build_tool_activation_request(
    *,
    payload: ToolLifecycleHandoffPayload,
    envelope: CapabilityHandoffEnvelope,
) -> ToolActivationRequest:
    identity_key = CapabilityIdentityKey.from_discovery_identity(
        envelope.selected_release.discovery,
    )
    if payload.capability_identity_key != identity_key:
        raise ValueError("handoff payload capability_identity_key must match envelope release")
    return ToolActivationRequest(payload=payload, selected_release=envelope.selected_release)


@dataclass
class MarketplaceToolHandoffConsumer:
    """ME-10 consumer: ME-RB4 Tool handler + reference Tool host lifecycle bridge."""

    lifecycle: ReferenceToolHostLifecycleService
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
        ) -> ToolActivationRequest:
            return build_tool_activation_request(
                payload=handoff_payload,
                envelope=envelope,
            )

        bridge = ToolMarketplaceAcquisitionBridge(
            self.lifecycle,
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
    lifecycle: ReferenceToolHostLifecycleService
    catalog_service: MarketplaceCatalogService
    orchestrator: MarketplaceDiscoveryHandoffOrchestrator
    handoff_consumer: MarketplaceToolHandoffConsumer
    delivery_admission: InMemoryCapabilityHandoffDeliveryAdmission
    delivery_service: CapabilityHandoffDeliveryService

    @classmethod
    def build(
        cls,
        *,
        listing_records: tuple[MarketplaceListingRecord, ...] | None = None,
        lifecycle: ReferenceToolHostLifecycleService | None = None,
    ) -> MarketplaceToolExecutionProofStack:
        resolved_lifecycle = lifecycle or ReferenceToolHostLifecycleService(
            host_profile_id=ME14_HOST_PROFILE_ID,
        )
        records = listing_records or (me14_default_listing_v1(),)
        catalog_service = build_marketplace_catalog_service(*records)
        handoff_consumer = MarketplaceToolHandoffConsumer(
            lifecycle=resolved_lifecycle,
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
            if item.identity.logical.logical_id == ME14_TOOL_LOGICAL_ID
        )
        return CapabilityIdentityKey.from_discovery_identity(entry.identity)

    async def execute_tool_via_declarative_boundary(
        self,
        *,
        tenant_id: str = "tenant-me14",
        run_seed: str = "me14-proof",
    ) -> tuple[str, str]:
        from intergrax.contracts.delegation_authority import ParentExecutionAuthority
        from intergrax.runtime.governance.active_execution_authority import (
            bind_active_execution_authority,
            reset_active_execution_authority,
        )
        from testing_support.builder import (
            canonical_governed_execution_scope,
            canonical_run_id_for_tests,
            canonical_task_id_for_tests,
        )
        from testing_support.catalog_declarative_invoker import (
            build_catalog_declarative_invoker_from_registry,
        )

        registry = self.lifecycle.registry_read()
        if not registry.has(ME14_TOOL_LOGICAL_ID):
            raise RuntimeError("tool not active in domain registry")
        invoker = build_catalog_declarative_invoker_from_registry(registry)
        canonical_run_id = canonical_run_id_for_tests(run_seed)
        canonical_task_id = canonical_task_id_for_tests(run_seed)
        invoker.bind_execution_identity(
            tenant_id=tenant_id,
            run_id=canonical_run_id,
            task_id=canonical_task_id,
            agent_id="me14.tool-proof.agent",
        )
        with canonical_governed_execution_scope(run_seed, bind_budget=True):
            authority_token = bind_active_execution_authority(
                ParentExecutionAuthority.unrestricted_root(),
            )
            try:
                result = await invoker.invoke(
                    tool_id=ME14_TOOL_LOGICAL_ID,
                    args={"message": "ping"},
                    idempotency_key=None,
                )
            finally:
                reset_active_execution_authority(authority_token)
        if result.status != "success":
            raise RuntimeError(f"tool invocation failed: {result}")
        output = result.output or {}
        execution_result = str(output.get("result", ""))
        return ME14_TOOL_LOGICAL_ID, execution_result

    def run_marketplace_tool_e2e(
        self,
        *,
        tenant_id: str | None = None,
        discovery_correlation_id: str = "discovery-corr-me14",
        selection_id: str = "selection-me14",
        handoff_id: str = "handoff-me14",
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
        execution_tool_id, execution_result = asyncio.run(
            self.execute_tool_via_declarative_boundary(),
        )
        expected = expected_output_for_release(selected_release)
        assert execution_result == expected
        return MarketplaceToolE2EProofEvidence(
            discovery_correlation_id=discovery_correlation_id,
            selection_id=selection_id,
            handoff_id=handoff_id,
            selected_release=selected_release,
            lifecycle_request_id=f"lifecycle:{handoff_id}",
            lifecycle_domain_reference=self.handoff_consumer.last_lifecycle_domain_reference or "",
            host_profile_id=self.lifecycle.host_profile_id,
            registry_tool_id=ME14_TOOL_LOGICAL_ID,
            activated_version_label=selected_release.version_label or "",
            activated_content_digest=selected_release.content_digest or "",
            execution_tool_id=execution_tool_id,
            execution_result=execution_result,
        )


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
    "build_marketplace_catalog_service",
    "build_tool_activation_request",
    "me14_default_listing_v1",
    "me14_listing_v2",
]
