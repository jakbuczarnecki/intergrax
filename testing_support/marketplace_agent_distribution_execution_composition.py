# © Artur Czarnecki. All rights reserved.

"""ME-13 reference composition: Marketplace → Agent Distribution → Execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from intergrax.agent_distribution.admin_models import (
    ActivateRuntimeRevisionRequest,
    BindAgentRequest,
    BuildApplicationRevisionRequest,
)
from intergrax.agent_distribution.catalog import (
    AgentDiscoveryCandidateIdentity,
    CatalogProviderKind,
    CatalogSourceIdentity,
)
from intergrax.agent_distribution.dependency import RepositoryDependencyDeclaration
from intergrax.agent_distribution.dynamic_acquisition import (
    CatalogSourceProviderRegistry,
    DynamicAgentAcquisitionInstallIntent,
    DynamicAgentAcquisitionRequest,
    DynamicAgentAcquisitionService,
)
from intergrax.agent_distribution.identity import AgentPackageCandidate
from intergrax.agent_distribution.runtime_revision import MaterializationTopology
from intergrax.agent_distribution.trust import (
    AgentInstallationTrustRecord,
    AgentQualificationEvidenceKind,
    AgentTrustEvidenceRef,
)
from intergrax.capability_catalog import (
    AvailabilityPreservingGovernanceEvaluator,
    FederatedCapabilityCatalog,
)
from intergrax.contracts.agent_distribution.marketplace_lifecycle_handoff import (
    AgentLifecycleHandoffPayload,
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
from intergrax.core.qualification import QualificationStatus
from intergrax.marketplace import (
    MarketplaceCapabilityCatalogSource,
    MarketplaceCatalogService,
    MarketplaceDiscoveryService,
)
from intergrax.marketplace.handoff import (
    AgentMarketplaceLifecycleHandoffHandler,
    LifecycleHandoffResolver,
    MarketplaceLifecycleHandoffService,
)
from intergrax.marketplace.handoff.adapters.agent_distribution_bridge import (
    AgentDistributionAcquisitionBridge,
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
from testing_support.agent_platform_admin_harness import admin_test_principal
from testing_support.canonical_agent_lifecycle_composition import (
    CanonicalAgentLifecycleProofStack,
    CanonicalLifecycleProofConfig,
    default_stage15_proof_config,
)
from testing_support.reference_production_acquisition_lifecycle_port import (
    ReferenceProductionAcquisitionLifecyclePort,
)

ME13_MARKETPLACE_LOGICAL_ID: Final = "agents.me13.canonical-ping"
ME13_CAPABILITY_SOURCE: Final = CapabilitySourceIdentity(
    source_id="official.intergrax.me13",
    source_kind=CapabilitySourceKind.OFFICIAL,
)
ME13_LISTING_ID: Final = "listing-me13-canonical-ping"


@dataclass(frozen=True, slots=True)
class MarketplaceAgentE2EProofEvidence:
    discovery_correlation_id: str
    selection_id: str
    handoff_id: str
    selected_release: CapabilityReleaseIdentity
    lifecycle_request_id: str
    installation_id: str
    application_binding_id: str
    runtime_revision_id: str
    traffic_serving_revision_id: str
    execution_agent_id: str
    execution_answer: str


def me13_lifecycle_proof_config() -> CanonicalLifecycleProofConfig:
    return default_stage15_proof_config(
        catalog_provider_kind=CatalogProviderKind.BUILTIN,
        catalog_source_id="builtin-me13",
    )


def _discovery_query() -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
    )


def _marketplace_listing_record(
    config: CanonicalLifecycleProofConfig,
    *,
    version_label: str | None = None,
    content_digest: str | None = None,
    tenant_id: str | None = None,
) -> MarketplaceListingRecord:
    visibility = None
    if tenant_id is not None:
        visibility = MarketplaceVisibility(
            scope=MarketplaceVisibilityScope.TENANT_PRIVATE,
            tenant_id=tenant_id,
        )
    return MarketplaceListingRecord(
        kind=CapabilityKind.AGENT,
        logical_id=ME13_MARKETPLACE_LOGICAL_ID,
        listing_id=ME13_LISTING_ID,
        display_label="ME-13 Canonical Ping",
        publisher="publisher:me13",
        version_label=version_label or config.package_version,
        content_digest=content_digest or config.package_digest,
        package_reference=config.distribution_package_id,
        visibility=visibility,
    )


def build_marketplace_catalog_service(
    *records: MarketplaceListingRecord,
) -> MarketplaceCatalogService:
    source = MarketplaceCapabilityCatalogSource(
        source=ME13_CAPABILITY_SOURCE,
        records=records,
    )
    catalog = FederatedCapabilityCatalog((source,))
    return MarketplaceCatalogService(catalog=catalog, marketplace_sources=(source,))


def agent_discovery_identity_from_release(
    *,
    release: CapabilityReleaseIdentity,
    catalog_source: CatalogSourceIdentity,
) -> AgentDiscoveryCandidateIdentity:
    package_reference = release.package_reference
    version_label = release.version_label
    if package_reference is None or not package_reference.strip():
        raise ValueError("selected release requires package_reference for agent acquisition")
    if version_label is None or not version_label.strip():
        raise ValueError("selected release requires version_label for exact acquisition")
    return AgentDiscoveryCandidateIdentity(
        source=catalog_source,
        package=AgentPackageCandidate(
            distribution_package_id=package_reference,
            package_version=version_label,
            package_digest=release.content_digest,
        ),
    )


def build_dynamic_agent_acquisition_request(
    *,
    payload: AgentLifecycleHandoffPayload,
    envelope: CapabilityHandoffEnvelope,
    config: CanonicalLifecycleProofConfig,
    catalog_source: CatalogSourceIdentity,
    operation_id: str,
) -> DynamicAgentAcquisitionRequest:
    identity_key = CapabilityIdentityKey.from_discovery_identity(
        envelope.selected_release.discovery,
    )
    if payload.capability_identity_key != identity_key:
        raise ValueError("handoff payload capability_identity_key must match envelope release")
    selected_identity = agent_discovery_identity_from_release(
        release=envelope.selected_release,
        catalog_source=catalog_source,
    )
    digest = selected_identity.package.package_digest or config.package_digest
    trust_record = AgentInstallationTrustRecord(
        qualification_status=QualificationStatus.PRODUCTION_QUALIFIED,
        package_digest=digest,
        publisher_identity_ref="publisher:me13",
        source_provider_id=catalog_source.catalog_source_id,
        trust_evidence_refs=(
            AgentTrustEvidenceRef(
                evidence_id="evidence:me13",
                kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
            ),
        ),
    )
    return DynamicAgentAcquisitionRequest(
        selected_identity=selected_identity,
        application_id=payload.application_id,
        application_environment_id=payload.application_environment_id,
        catalog_entry_id=payload.catalog_entry_id,
        install=DynamicAgentAcquisitionInstallIntent(
            mutation_id=f"mut-me13-install:{config.installation_id}",
            installation_id=config.installation_id,
            installation_slot_id=config.installation_slot_id,
            artifact_store_ref=f"store://artifacts/{config.installation_id}",
            trust_record=trust_record,
            agent_project_metadata_ref=config.metadata_ref,
        ),
        bind=BindAgentRequest(
            mutation_id=f"mut-me13-bind:{config.application_binding_id}",
            application_binding_id=config.application_binding_id,
            logical_agent_id=config.logical_agent_id,
            installation_slot_id=config.installation_slot_id,
            factory_reference=config.factory_reference,
            enablement=True,
        ),
        build=BuildApplicationRevisionRequest(
            mutation_id=f"mut-me13-build:{config.revision_id}",
            runtime_revision_id=config.revision_id,
            application_release_id="rel-me13",
            platform_version="0.1.0",
            python_version="3.12",
            source_context_root="/tmp/src",
            output_root="/tmp/out",
            application_source_root=f"applications/{config.application_id}",
            materialization_topology=MaterializationTopology.VENV_BUNDLE,
            repository_declaration=RepositoryDependencyDeclaration(
                application_release_id="rel-me13",
                direct_dependencies=(),
            ),
            resolver_algorithm_id="intergrax.me13-resolver",
            resolver_algorithm_version="1.0.0",
        ),
        activate=ActivateRuntimeRevisionRequest(
            mutation_id=f"mut-me13-activate:{config.revision_id}",
            runtime_revision_id=config.revision_id,
            artifact_locator="pending://me13",
            expected_artifact_digest=config.package_digest,
            expected_serving_pointer_revision=0,
            expected_prior_traffic_revision_id=None,
        ),
    )


@dataclass
class MarketplaceAgentDistributionHandoffConsumer:
    """ME-10 consumer that delegates to ME-RB4 + Agent Distribution acquisition bridge."""

    acquisition: DynamicAgentAcquisitionService
    lifecycle_config: CanonicalLifecycleProofConfig
    agent_catalog_source: CatalogSourceIdentity
    _delivered_handoffs: list[str] = field(default_factory=list)
    last_envelope: CapabilityHandoffEnvelope | None = field(default=None, init=False)

    @property
    def consumer_id(self) -> str:
        return "me13.agent_distribution.handoff_consumer"

    def consume(self, envelope: CapabilityHandoffEnvelope) -> None:
        identity_key = CapabilityIdentityKey.from_discovery_identity(
            envelope.selected_release.discovery,
        )
        operation_id = envelope.handoff_id
        payload = AgentLifecycleHandoffPayload(
            operation_id=operation_id,
            application_id=self.lifecycle_config.application_id,
            application_environment_id=self.lifecycle_config.environment_id,
            catalog_entry_id=self.lifecycle_config.catalog_entry_id,
            capability_identity_key=identity_key,
        )
        lifecycle_request = lifecycle_handoff_request_from_envelope(
            envelope,
            request_id=f"lifecycle:{envelope.handoff_id}",
            intent=MarketplaceLifecycleHandoffIntent.REQUEST_LIFECYCLE,
            domain_payload=MarketplaceLifecycleDomainPayload(agent=payload),
            correlation_id=envelope.discovery_correlation_id,
        )

        def request_factory(
            handoff_payload: AgentLifecycleHandoffPayload,
        ) -> DynamicAgentAcquisitionRequest:
            return build_dynamic_agent_acquisition_request(
                payload=handoff_payload,
                envelope=envelope,
                config=self.lifecycle_config,
                catalog_source=self.agent_catalog_source,
                operation_id=operation_id,
            )

        bridge = AgentDistributionAcquisitionBridge(
            self.acquisition,
            request_factory,
            principal=admin_test_principal(),
        )
        handler = AgentMarketplaceLifecycleHandoffHandler(bridge)
        resolver = LifecycleHandoffResolver({CapabilityKind.AGENT: handler})
        outcome = MarketplaceLifecycleHandoffService(resolver).handoff(lifecycle_request)
        if outcome.status is not MarketplaceLifecycleHandoffStatus.ACCEPTED:
            raise CapabilityHandoffConsumerError(
                f"agent distribution rejected marketplace handoff: {outcome.reason_detail}",
            )
        self.last_envelope = envelope
        self._delivered_handoffs.append(envelope.handoff_id)


@dataclass(frozen=True, slots=True)
class MarketplaceAgentDistributionExecutionProofStack:
    lifecycle_stack: CanonicalAgentLifecycleProofStack
    catalog_service: MarketplaceCatalogService
    orchestrator: MarketplaceDiscoveryHandoffOrchestrator
    handoff_consumer: MarketplaceAgentDistributionHandoffConsumer
    acquisition: DynamicAgentAcquisitionService
    delivery_admission: InMemoryCapabilityHandoffDeliveryAdmission
    delivery_service: CapabilityHandoffDeliveryService

    @classmethod
    def build(
        cls,
        tmp_path: Path,
        *,
        lifecycle_config: CanonicalLifecycleProofConfig | None = None,
        listing_records: tuple[MarketplaceListingRecord, ...] | None = None,
    ) -> MarketplaceAgentDistributionExecutionProofStack:
        resolved_config = lifecycle_config or me13_lifecycle_proof_config()
        lifecycle_stack = CanonicalAgentLifecycleProofStack.build(
            tmp_path,
            resolved_config,
        )
        catalog_entry = lifecycle_stack.discover_catalog_entry()
        agent_catalog_source = catalog_entry.catalog_source
        records = listing_records or (_marketplace_listing_record(resolved_config),)
        catalog_service = build_marketplace_catalog_service(*records)
        lifecycle_port = ReferenceProductionAcquisitionLifecyclePort(
            admin=lifecycle_stack.admin,
            launcher=lifecycle_stack.launcher,
            manifest=lifecycle_stack.manifest,
            registry_projection_authority=(
                lifecycle_stack.composition.agent_platform_runtime.registry_projection_authority
            ),
            principal=lifecycle_stack.governance.principal,
        )
        acquisition = DynamicAgentAcquisitionService(
            catalog_registry=CatalogSourceProviderRegistry(
                {
                    lifecycle_stack.catalog_provider.catalog_source_id: (
                        lifecycle_stack.catalog_provider
                    ),
                },
            ),
            lifecycle=lifecycle_port,
        )
        handoff_consumer = MarketplaceAgentDistributionHandoffConsumer(
            acquisition=acquisition,
            lifecycle_config=resolved_config,
            agent_catalog_source=agent_catalog_source,
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
            lifecycle_stack=lifecycle_stack,
            catalog_service=catalog_service,
            orchestrator=orchestrator,
            handoff_consumer=handoff_consumer,
            acquisition=acquisition,
            delivery_admission=delivery_admission,
            delivery_service=delivery,
        )

    def marketplace_identity_key(self) -> CapabilityIdentityKey:
        snapshot = self.catalog_service._catalog.snapshot()
        entry = next(
            item
            for item in snapshot.entries
            if item.identity.logical.logical_id == ME13_MARKETPLACE_LOGICAL_ID
        )
        return CapabilityIdentityKey.from_discovery_identity(entry.identity)

    def run_marketplace_agent_e2e(
        self,
        *,
        tenant_id: str | None = None,
        discovery_correlation_id: str = "discovery-corr-me13",
        selection_id: str = "selection-me13",
        handoff_id: str = "handoff-me13",
    ) -> MarketplaceAgentE2EProofEvidence:
        identity_key = self.marketplace_identity_key()
        self.orchestrator.execute_explicit_selection_handoff(
            discovery_query=_discovery_query(),
            marketplace_query_context=MarketplaceQueryContext(tenant_id=tenant_id),
            selected_identity_key=identity_key,
            consumer_target=CapabilityHandoffConsumerTarget.AGENT_DOMAIN,
            selector_id="operator.me13.explicit",
            discovery_correlation_id=discovery_correlation_id,
            selection_id=selection_id,
            handoff_id=handoff_id,
        )
        stack = self.lifecycle_stack
        config = stack.config
        serving = stack.admin.inspect_serving(
            application_id=config.application_id,
            application_environment_id=config.environment_id,
        )
        assert serving.traffic_serving_revision_id == config.revision_id
        registry = stack.resolve_registry_read()
        assert registry.has(config.logical_agent_id)
        materialization = (
            stack.composition.agent_platform_runtime.stores.materialization_store.get_by_revision(
                config.revision_id,
            )
        )
        assert materialization is not None
        execution_agent_id, execution_answer = asyncio.run(stack.execute_canonical())
        catalog_entry = next(
            entry
            for entry in self.catalog_service._catalog.snapshot().entries
            if entry.identity.logical.logical_id == ME13_MARKETPLACE_LOGICAL_ID
        )
        envelope_release = CapabilityReleaseIdentity.from_catalog_entry(catalog_entry)
        return MarketplaceAgentE2EProofEvidence(
            discovery_correlation_id=discovery_correlation_id,
            selection_id=selection_id,
            handoff_id=handoff_id,
            selected_release=envelope_release,
            lifecycle_request_id=f"lifecycle:{handoff_id}",
            installation_id=config.installation_id,
            application_binding_id=config.application_binding_id,
            runtime_revision_id=config.revision_id,
            traffic_serving_revision_id=serving.traffic_serving_revision_id or config.revision_id,
            execution_agent_id=execution_agent_id,
            execution_answer=execution_answer,
        )


__all__ = [
    "ME13_CAPABILITY_SOURCE",
    "ME13_LISTING_ID",
    "ME13_MARKETPLACE_LOGICAL_ID",
    "MarketplaceAgentDistributionExecutionProofStack",
    "MarketplaceAgentDistributionHandoffConsumer",
    "MarketplaceAgentE2EProofEvidence",
    "agent_discovery_identity_from_release",
    "build_dynamic_agent_acquisition_request",
    "build_marketplace_catalog_service",
    "me13_lifecycle_proof_config",
]
