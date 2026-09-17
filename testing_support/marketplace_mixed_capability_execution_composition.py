# © Artur Czarnecki. All rights reserved.

"""ME-16 reference composition: common Marketplace → Agent + Tool + Skill → Execution."""

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
    CapabilityHandoffConsumerTarget,
    CapabilityHandoffEnvelope,
)
from intergrax.contracts.marketplace.listing_record import MarketplaceListingRecord
from intergrax.contracts.marketplace.query_context import MarketplaceQueryContext
from intergrax.contracts.marketplace.visibility import (
    MarketplaceVisibility,
    MarketplaceVisibilityScope,
)
from intergrax.marketplace import (
    MarketplaceCapabilityCatalogSource,
    MarketplaceCatalogService,
    MarketplaceDiscoveryService,
)
from intergrax.marketplace.handoff_traceability import (
    CapabilityHandoffDeliveryService,
    InMemoryCapabilityHandoffDeliveryAdmission,
    InMemoryCapabilityHandoffTraceEvidenceConsumer,
    MarketplaceDiscoveryHandoffOrchestrator,
)
from intergrax.skills.host_lifecycle import SkillHostLifecycleService
from intergrax.tools.host_lifecycle import ToolHostLifecycleService
from testing_support.canonical_agent_lifecycle_composition import (
    CanonicalAgentLifecycleProofStack,
    me16_mixed_lifecycle_proof_config,
)
from testing_support.canonical_me14_echo_tool import (
    ME14_TOOL_LOGICAL_ID,
)
from testing_support.canonical_me15_reference_skill import (
    ME15_SKILL_LOGICAL_ID,
)
from testing_support.canonical_me16_mixed_agent import (
    ME16_LISTING_ID,
    ME16_MARKETPLACE_LOGICAL_ID,
    expected_mixed_output_for_releases,
)
from testing_support.marketplace_agent_distribution_execution_composition import (
    MarketplaceAgentDistributionHandoffConsumer,
)
from testing_support.marketplace_skill_composition import MarketplaceSkillHandoffConsumer
from testing_support.marketplace_tool_execution_composition import (
    MarketplaceToolHandoffConsumer,
)
from testing_support.me16_mixed_harness_execution import (
    execute_me16_mixed_via_host_execution_engine,
)
from intergrax.agent_distribution.dynamic_acquisition import (
    CatalogSourceProviderRegistry,
    DynamicAgentAcquisitionService,
)
from intergrax.skills.catalog import SkillCatalogProviderRegistry
from intergrax.skills.dynamic_acquisition import DynamicSkillAcquisitionService
from intergrax.tools.catalog import ToolCatalogProviderRegistry
from intergrax.tools.dynamic_acquisition import DynamicToolAcquisitionService
from testing_support.me14_tool_activation_materializer import Me14ToolHostActivationMaterializer
from testing_support.me14_tool_catalog_provider import Me14ToolCatalogProvider
from testing_support.me15_skill_binding_materializer import Me15SkillHostBindingMaterializer
from testing_support.me15_skill_catalog_provider import Me15SkillCatalogProvider
from testing_support.reference_production_acquisition_lifecycle_port import (
    ReferenceProductionAcquisitionLifecyclePort,
)

ME16_AGENT_CAPABILITY_SOURCE: Final = CapabilitySourceIdentity(
    source_id="official.intergrax.me16",
    source_kind=CapabilitySourceKind.OFFICIAL,
)
ME16_HOST_PROFILE_TOOL: Final = "host-profile-me16-tool"
ME16_HOST_PROFILE_SKILL: Final = "host-profile-me16-skill"


class MixedCapabilityCompositionNotReadyError(RuntimeError):
    """Mixed Agent+Tool+Skill composition is not ready for canonical execution."""

    def __init__(self, readiness: MixedCapabilityCompositionReadiness) -> None:
        self.readiness = readiness
        super().__init__(
            "mixed capability composition not ready: "
            f"agent={readiness.agent_ready} tool={readiness.tool_ready} "
            f"skill={readiness.skill_ready}",
        )


@dataclass(frozen=True, slots=True)
class MixedCapabilityCompositionReadiness:
    agent_ready: bool
    tool_ready: bool
    skill_ready: bool

    @property
    def execution_allowed(self) -> bool:
        return self.agent_ready and self.tool_ready and self.skill_ready


@dataclass(frozen=True, slots=True)
class MarketplaceMixedCapabilityE2EProofEvidence:
    mixed_operation_id: str
    discovery_correlation_id: str
    agent_selection_id: str
    tool_selection_id: str
    skill_selection_id: str
    agent_handoff_id: str
    tool_handoff_id: str
    skill_handoff_id: str
    agent_selected_release: CapabilityReleaseIdentity
    tool_selected_release: CapabilityReleaseIdentity
    skill_selected_release: CapabilityReleaseIdentity
    agent_envelope: CapabilityHandoffEnvelope
    tool_envelope: CapabilityHandoffEnvelope
    skill_envelope: CapabilityHandoffEnvelope
    execution_agent_id: str
    execution_answer: str
    execution_task_id: str | None
    agent_installation_id: str
    tool_activation_version: str
    skill_bound_version: str
    skill_snapshot_digest: str


def _discovery_query() -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
    )


def me16_agent_listing_v1(
    config: object,
    *,
    tenant_id: str | None = None,
) -> MarketplaceListingRecord:
    from testing_support.canonical_agent_lifecycle_composition import (
        CanonicalLifecycleProofConfig,
    )

    assert isinstance(config, CanonicalLifecycleProofConfig)
    visibility = None
    if tenant_id is not None:
        visibility = MarketplaceVisibility(
            scope=MarketplaceVisibilityScope.TENANT_PRIVATE,
            tenant_id=tenant_id,
        )
    return MarketplaceListingRecord(
        kind=CapabilityKind.AGENT,
        logical_id=ME16_MARKETPLACE_LOGICAL_ID,
        listing_id=ME16_LISTING_ID,
        display_label="ME-16 Mixed Worker",
        publisher="publisher:me16",
        version_label=config.package_version,
        content_digest=config.package_digest,
        package_reference=config.distribution_package_id,
        visibility=visibility,
    )


def me16_agent_listing_v2(config: object) -> MarketplaceListingRecord:
    from testing_support.canonical_agent_lifecycle_composition import (
        CanonicalLifecycleProofConfig,
    )

    assert isinstance(config, CanonicalLifecycleProofConfig)
    return MarketplaceListingRecord(
        kind=CapabilityKind.AGENT,
        logical_id=ME16_MARKETPLACE_LOGICAL_ID,
        listing_id=ME16_LISTING_ID,
        display_label="ME-16 Mixed Worker v2",
        publisher="publisher:me16",
        version_label="2.0.0",
        content_digest="sha256:" + ("e" * 64),
        package_reference=config.distribution_package_id,
    )


def build_mixed_marketplace_catalog_service(
    *,
    agent_records: tuple[MarketplaceListingRecord, ...],
    tool_records: tuple[MarketplaceListingRecord, ...],
    skill_records: tuple[MarketplaceListingRecord, ...],
    tool_catalog_source_id: str,
    skill_catalog_source_id: str,
) -> MarketplaceCatalogService:
    agent_source = MarketplaceCapabilityCatalogSource(
        source=ME16_AGENT_CAPABILITY_SOURCE,
        records=agent_records,
    )
    tool_source = MarketplaceCapabilityCatalogSource(
        source=CapabilitySourceIdentity(
            source_id=tool_catalog_source_id,
            source_kind=CapabilitySourceKind.OFFICIAL,
        ),
        records=tool_records,
    )
    skill_source = MarketplaceCapabilityCatalogSource(
        source=CapabilitySourceIdentity(
            source_id=skill_catalog_source_id,
            source_kind=CapabilitySourceKind.OFFICIAL,
        ),
        records=skill_records,
    )
    catalog = FederatedCapabilityCatalog((agent_source, tool_source, skill_source))
    return MarketplaceCatalogService(
        catalog=catalog,
        marketplace_sources=(agent_source, tool_source, skill_source),
    )


@dataclass
class MarketplaceMixedHandoffConsumer:
    """Routes envelopes to vertical consumers — not a lifecycle authority."""

    agent: MarketplaceAgentDistributionHandoffConsumer
    tool: MarketplaceToolHandoffConsumer
    skill: MarketplaceSkillHandoffConsumer
    _delivered: list[str] = field(default_factory=list)

    @property
    def consumer_id(self) -> str:
        return "me16.mixed.handoff_consumer"

    def consume(self, envelope: CapabilityHandoffEnvelope) -> None:
        target = envelope.consumer_target
        if target is CapabilityHandoffConsumerTarget.AGENT_DOMAIN:
            self.agent.consume(envelope)
        elif target is CapabilityHandoffConsumerTarget.TOOL_DOMAIN:
            self.tool.consume(envelope)
        elif target is CapabilityHandoffConsumerTarget.SKILL_DOMAIN:
            self.skill.consume(envelope)
        else:
            raise CapabilityHandoffConsumerError(f"unsupported consumer target: {target}")
        self._delivered.append(envelope.handoff_id)


@dataclass(frozen=True, slots=True)
class MarketplaceMixedCapabilityProofStack:
    mixed_operation_id: str
    lifecycle_config: object
    agent_stack: CanonicalAgentLifecycleProofStack
    tool_lifecycle: ToolHostLifecycleService
    skill_lifecycle: SkillHostLifecycleService
    catalog_service: MarketplaceCatalogService
    orchestrator: MarketplaceDiscoveryHandoffOrchestrator
    mixed_consumer: MarketplaceMixedHandoffConsumer
    delivery_admission: InMemoryCapabilityHandoffDeliveryAdmission
    delivery_service: CapabilityHandoffDeliveryService
    agent_acquisition: DynamicAgentAcquisitionService
    tool_acquisition: DynamicToolAcquisitionService
    skill_acquisition: DynamicSkillAcquisitionService

    @classmethod
    def build(
        cls,
        tmp_path: Path,
        *,
        tool_catalog_provider: Me14ToolCatalogProvider | None = None,
        skill_catalog_provider: Me15SkillCatalogProvider | None = None,
        agent_listing_records: tuple[MarketplaceListingRecord, ...] | None = None,
        tool_listing_records: tuple[MarketplaceListingRecord, ...] | None = None,
        skill_listing_records: tuple[MarketplaceListingRecord, ...] | None = None,
        mixed_operation_id: str = "mixed-op-me16",
    ) -> MarketplaceMixedCapabilityProofStack:
        lifecycle_config = me16_mixed_lifecycle_proof_config()
        agent_stack = CanonicalAgentLifecycleProofStack.build(tmp_path, lifecycle_config)
        catalog_entry = agent_stack.discover_catalog_entry()
        agent_catalog_source = catalog_entry.catalog_source

        tool_provider = tool_catalog_provider or Me14ToolCatalogProvider()
        skill_provider = skill_catalog_provider or Me15SkillCatalogProvider()
        tool_lifecycle = ToolHostLifecycleService(host_profile_id=ME16_HOST_PROFILE_TOOL)
        skill_lifecycle = SkillHostLifecycleService(host_profile_id=ME16_HOST_PROFILE_SKILL)
        tool_materializer = Me14ToolHostActivationMaterializer(
            tool_lifecycle.registry,
            catalog_source_id=tool_provider.catalog_source_id,
        )
        tool_catalog_registry: dict[str, object] = {
            tool_provider.catalog_source_id: tool_provider,
        }
        if tool_catalog_provider is not None:
            default_tool_provider = Me14ToolCatalogProvider()
            tool_catalog_registry[default_tool_provider.catalog_source_id] = (
                default_tool_provider
            )
        tool_acquisition = DynamicToolAcquisitionService(
            catalog_registry=ToolCatalogProviderRegistry(tool_catalog_registry),
            activation=tool_lifecycle,
            materializer=tool_materializer,
        )
        skill_materializer = Me15SkillHostBindingMaterializer(
            skill_lifecycle.registry,
            catalog_source_id=skill_provider.catalog_source_id,
        )
        skill_acquisition = DynamicSkillAcquisitionService(
            catalog_registry=SkillCatalogProviderRegistry(
                {skill_provider.catalog_source_id: skill_provider},
            ),
            binding=skill_lifecycle,
            materializer=skill_materializer,
        )
        lifecycle_port = ReferenceProductionAcquisitionLifecyclePort(
            admin=agent_stack.admin,
            launcher=agent_stack.launcher,
            manifest=agent_stack.manifest,
            registry_projection_authority=(
                agent_stack.composition.agent_platform_runtime.registry_projection_authority
            ),
            principal=agent_stack.governance.principal,
        )
        agent_acquisition = DynamicAgentAcquisitionService(
            catalog_registry=CatalogSourceProviderRegistry(
                {
                    agent_stack.catalog_provider.catalog_source_id: (
                        agent_stack.catalog_provider
                    ),
                },
            ),
            lifecycle=lifecycle_port,
        )
        agent_consumer = MarketplaceAgentDistributionHandoffConsumer(
            acquisition=agent_acquisition,
            lifecycle_config=lifecycle_config,
            agent_catalog_source=agent_catalog_source,
        )
        tool_consumer = MarketplaceToolHandoffConsumer(
            acquisition=tool_acquisition,
            host_profile_id=tool_lifecycle.host_profile_id,
        )
        skill_consumer = MarketplaceSkillHandoffConsumer(
            acquisition=skill_acquisition,
            host_profile_id=skill_lifecycle.host_profile_id,
        )
        mixed_consumer = MarketplaceMixedHandoffConsumer(
            agent=agent_consumer,
            tool=tool_consumer,
            skill=skill_consumer,
        )
        delivery_admission = InMemoryCapabilityHandoffDeliveryAdmission()
        delivery = CapabilityHandoffDeliveryService(
            consumer=mixed_consumer,
            delivery_admission=delivery_admission,
            trace_evidence_consumer=InMemoryCapabilityHandoffTraceEvidenceConsumer(),
        )
        from testing_support.marketplace_tool_execution_composition import me14_default_listing_v1
        from testing_support.marketplace_skill_composition import me15_default_listing_v1

        agent_records = agent_listing_records or (me16_agent_listing_v1(lifecycle_config),)
        tool_records = tool_listing_records or (me14_default_listing_v1(),)
        skill_records = skill_listing_records or (me15_default_listing_v1(),)
        catalog_service = build_mixed_marketplace_catalog_service(
            agent_records=agent_records,
            tool_records=tool_records,
            skill_records=skill_records,
            tool_catalog_source_id=tool_provider.catalog_source_id,
            skill_catalog_source_id=skill_provider.catalog_source_id,
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
            mixed_operation_id=mixed_operation_id,
            lifecycle_config=lifecycle_config,
            agent_stack=agent_stack,
            tool_lifecycle=tool_lifecycle,
            skill_lifecycle=skill_lifecycle,
            catalog_service=catalog_service,
            orchestrator=orchestrator,
            mixed_consumer=mixed_consumer,
            delivery_admission=delivery_admission,
            delivery_service=delivery,
            agent_acquisition=agent_acquisition,
            tool_acquisition=tool_acquisition,
            skill_acquisition=skill_acquisition,
        )

    def readiness(self) -> MixedCapabilityCompositionReadiness:
        config = self.lifecycle_config
        serving = self.agent_stack.admin.inspect_serving(
            application_id=config.application_id,
            application_environment_id=config.environment_id,
        )
        serving_ready = serving.traffic_serving_revision_id == config.revision_id
        agent_handoff = self.mixed_consumer.agent.last_envelope is not None
        tool_handoff = self.mixed_consumer.tool.last_envelope is not None
        skill_handoff = self.mixed_consumer.skill.last_envelope is not None
        return MixedCapabilityCompositionReadiness(
            agent_ready=(
                agent_handoff
                and serving_ready
                and self.agent_stack.resolve_registry_read().has(
                    config.logical_agent_id,
                )
            ),
            tool_ready=tool_handoff and self.tool_lifecycle.is_active(ME14_TOOL_LOGICAL_ID),
            skill_ready=skill_handoff and self.skill_lifecycle.is_bound(ME15_SKILL_LOGICAL_ID),
        )

    def assert_execution_readiness(self) -> MixedCapabilityCompositionReadiness:
        readiness = self.readiness()
        if not readiness.execution_allowed:
            raise MixedCapabilityCompositionNotReadyError(readiness)
        return readiness

    def _identity_key_for(self, logical_id: str, kind: CapabilityKind) -> CapabilityIdentityKey:
        snapshot = self.catalog_service._catalog.snapshot()
        entry = next(
            item
            for item in snapshot.entries
            if item.identity.logical.logical_id == logical_id
            and item.identity.logical.kind == kind
        )
        return CapabilityIdentityKey.from_discovery_identity(entry.identity)

    def handoff_agent(
        self,
        *,
        discovery_correlation_id: str,
        selection_id: str,
        handoff_id: str,
        tenant_id: str | None = None,
    ) -> CapabilityHandoffEnvelope:
        self.orchestrator.execute_explicit_selection_handoff(
            discovery_query=_discovery_query(),
            marketplace_query_context=MarketplaceQueryContext(tenant_id=tenant_id),
            selected_identity_key=self._identity_key_for(
                ME16_MARKETPLACE_LOGICAL_ID,
                CapabilityKind.AGENT,
            ),
            consumer_target=CapabilityHandoffConsumerTarget.AGENT_DOMAIN,
            selector_id="operator.me16.agent",
            discovery_correlation_id=discovery_correlation_id,
            selection_id=selection_id,
            handoff_id=handoff_id,
        )
        envelope = self.mixed_consumer.agent.last_envelope
        assert envelope is not None
        return envelope

    def handoff_tool(
        self,
        *,
        discovery_correlation_id: str,
        selection_id: str,
        handoff_id: str,
        tenant_id: str | None = None,
    ) -> CapabilityHandoffEnvelope:
        self.orchestrator.execute_explicit_selection_handoff(
            discovery_query=_discovery_query(),
            marketplace_query_context=MarketplaceQueryContext(tenant_id=tenant_id),
            selected_identity_key=self._identity_key_for(
                ME14_TOOL_LOGICAL_ID,
                CapabilityKind.TOOL,
            ),
            consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
            selector_id="operator.me16.tool",
            discovery_correlation_id=discovery_correlation_id,
            selection_id=selection_id,
            handoff_id=handoff_id,
        )
        envelope = self.mixed_consumer.tool.last_envelope
        assert envelope is not None
        return envelope

    def handoff_skill(
        self,
        *,
        discovery_correlation_id: str,
        selection_id: str,
        handoff_id: str,
        tenant_id: str | None = None,
    ) -> CapabilityHandoffEnvelope:
        self.orchestrator.execute_explicit_selection_handoff(
            discovery_query=_discovery_query(),
            marketplace_query_context=MarketplaceQueryContext(tenant_id=tenant_id),
            selected_identity_key=self._identity_key_for(
                ME15_SKILL_LOGICAL_ID,
                CapabilityKind.SKILL,
            ),
            consumer_target=CapabilityHandoffConsumerTarget.SKILL_DOMAIN,
            selector_id="operator.me16.skill",
            discovery_correlation_id=discovery_correlation_id,
            selection_id=selection_id,
            handoff_id=handoff_id,
        )
        envelope = self.mixed_consumer.skill.last_envelope
        assert envelope is not None
        return envelope

    def run_all_handoffs(
        self,
        *,
        discovery_correlation_id: str = "discovery-corr-me16",
        agent_selection_id: str = "selection-me16-agent",
        tool_selection_id: str = "selection-me16-tool",
        skill_selection_id: str = "selection-me16-skill",
        agent_handoff_id: str = "handoff-me16-agent",
        tool_handoff_id: str = "handoff-me16-tool",
        skill_handoff_id: str = "handoff-me16-skill",
        tenant_id: str | None = None,
    ) -> tuple[
        CapabilityHandoffEnvelope,
        CapabilityHandoffEnvelope,
        CapabilityHandoffEnvelope,
    ]:
        agent_env = self.handoff_agent(
            discovery_correlation_id=discovery_correlation_id,
            selection_id=agent_selection_id,
            handoff_id=agent_handoff_id,
            tenant_id=tenant_id,
        )
        tool_env = self.handoff_tool(
            discovery_correlation_id=discovery_correlation_id,
            selection_id=tool_selection_id,
            handoff_id=tool_handoff_id,
            tenant_id=tenant_id,
        )
        skill_env = self.handoff_skill(
            discovery_correlation_id=discovery_correlation_id,
            selection_id=skill_selection_id,
            handoff_id=skill_handoff_id,
            tenant_id=tenant_id,
        )
        return agent_env, tool_env, skill_env

    def run_marketplace_mixed_e2e(
        self,
        *,
        execution_tmp_path: Path,
        discovery_correlation_id: str = "discovery-corr-me16",
        agent_selection_id: str = "selection-me16-agent",
        tool_selection_id: str = "selection-me16-tool",
        skill_selection_id: str = "selection-me16-skill",
        agent_handoff_id: str = "handoff-me16-agent",
        tool_handoff_id: str = "handoff-me16-tool",
        skill_handoff_id: str = "handoff-me16-skill",
        tenant_id: str | None = None,
    ) -> MarketplaceMixedCapabilityE2EProofEvidence:
        if self.readiness().execution_allowed:
            raise RuntimeError("mixed stack must not be ready before marketplace handoffs")
        agent_env, tool_env, skill_env = self.run_all_handoffs(
            discovery_correlation_id=discovery_correlation_id,
            agent_selection_id=agent_selection_id,
            tool_selection_id=tool_selection_id,
            skill_selection_id=skill_selection_id,
            agent_handoff_id=agent_handoff_id,
            tool_handoff_id=tool_handoff_id,
            skill_handoff_id=skill_handoff_id,
            tenant_id=tenant_id,
        )
        readiness = self.readiness()
        if not readiness.execution_allowed:
            raise RuntimeError("mixed composition not ready after handoffs")
        from intergrax.skills.execution_binding import resolve_skill_composition_from_profile

        composition = resolve_skill_composition_from_profile(
            self.skill_lifecycle.skill_profile,
            skill_registry=self.skill_lifecycle.registry,
        )
        tool_registry = self.tool_lifecycle.registry_read()
        execution_agent_id, execution_answer, execution_task_id = asyncio.run(
            execute_me16_mixed_via_host_execution_engine(
                agent_stack=self.agent_stack,
                tool_registry=tool_registry,
                skill_lifecycle=self.skill_lifecycle,
                tmp_path=execution_tmp_path,
            ),
        )
        config = self.lifecycle_config
        expected = expected_mixed_output_for_releases(
            skill_release=skill_env.selected_release,
            tool_release=tool_env.selected_release,
        )
        assert execution_answer == expected
        tool_activation = self.tool_lifecycle.activation_metadata(ME14_TOOL_LOGICAL_ID)
        skill_binding = self.skill_lifecycle.binding_metadata(ME15_SKILL_LOGICAL_ID)
        assert tool_activation is not None and skill_binding is not None
        return MarketplaceMixedCapabilityE2EProofEvidence(
            mixed_operation_id=self.mixed_operation_id,
            discovery_correlation_id=discovery_correlation_id,
            agent_selection_id=agent_selection_id,
            tool_selection_id=tool_selection_id,
            skill_selection_id=skill_selection_id,
            agent_handoff_id=agent_handoff_id,
            tool_handoff_id=tool_handoff_id,
            skill_handoff_id=skill_handoff_id,
            agent_selected_release=agent_env.selected_release,
            tool_selected_release=tool_env.selected_release,
            skill_selected_release=skill_env.selected_release,
            agent_envelope=agent_env,
            tool_envelope=tool_env,
            skill_envelope=skill_env,
            execution_agent_id=execution_agent_id,
            execution_answer=execution_answer,
            execution_task_id=execution_task_id,
            agent_installation_id=config.installation_id,
            tool_activation_version=tool_activation.version_label,
            skill_bound_version=skill_binding.version_label,
            skill_snapshot_digest=composition.pack.snapshot_digest,
        )

    async def try_execute_when_not_ready(self, tmp_path: Path) -> None:
        if self.readiness().execution_allowed:
            raise AssertionError("expected partial readiness")
        self.assert_execution_readiness()


__all__ = [
    "ME16_AGENT_CAPABILITY_SOURCE",
    "MarketplaceMixedCapabilityE2EProofEvidence",
    "MarketplaceMixedCapabilityProofStack",
    "MarketplaceMixedHandoffConsumer",
    "MixedCapabilityCompositionNotReadyError",
    "MixedCapabilityCompositionReadiness",
    "build_mixed_marketplace_catalog_service",
    "me16_agent_listing_v1",
    "me16_agent_listing_v2",
]
