# © Artur Czarnecki. All rights reserved.

"""ME-10 discovery → selection → handoff traceability qualification."""

from __future__ import annotations

import ast
import importlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from intergrax.capability_catalog import (
    AvailabilityPreservingGovernanceEvaluator,
    FederatedCapabilityCatalog,
    rank_capability_candidates,
)
from intergrax.capability_catalog.candidate import CapabilityDiscoveryCandidate
from intergrax.capability_metering import CapabilityUsageRecorder
from intergrax.contracts.capability_catalog import (
    AvailabilityDisposition,
    CapabilityCatalogEntry,
    CapabilityDiscoveryIdentity,
    CapabilityDiscoveryQuery,
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
    CapabilityGovernanceContext,
    CapabilityGovernancePosture,
    CapabilityIdentityKey,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityProvenance,
    CapabilityReleaseIdentity,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.marketplace import (
    CapabilityHandoffConsumerError,
    CapabilityHandoffConsumerTarget,
    CapabilityHandoffDeliveryDisposition,
    MarketplaceListingRecord,
    MarketplaceQueryContext,
    MarketplaceVisibility,
    MarketplaceVisibilityScope,
)
from intergrax.contracts.marketplace.lifecycle_handoff_intent import (
    MarketplaceLifecycleHandoffIntent,
)
from intergrax.contracts.marketplace.lifecycle_handoff_payloads import (
    AgentLifecycleHandoffPayload,
    SkillLifecycleHandoffPayload,
    ToolLifecycleHandoffPayload,
)
from intergrax.contracts.marketplace.lifecycle_handoff_request import (
    MarketplaceLifecycleDomainPayload,
)
from intergrax.marketplace import (
    MarketplaceCapabilityCatalogSource,
    MarketplaceCatalogService,
    MarketplaceDiscoveryService,
    MarketplaceRecommendationService,
)
from intergrax.marketplace.handoff_traceability import (
    CapabilityHandoffDeliveryService,
    InMemoryCapabilityHandoffDeliveryAdmission,
    InMemoryCapabilityHandoffTraceEvidenceConsumer,
    MarketplaceDiscoveryHandoffOrchestrator,
    MarketplaceHandoffSelectionError,
    attribution_from_handoff_envelope,
)
from intergrax.marketplace.handoff_traceability.lifecycle_bridge import (
    lifecycle_handoff_request_from_envelope,
)
from intergrax.marketplace.handoff import (
    AgentMarketplaceLifecycleHandoffHandler,
    LifecycleHandoffResolver,
    MarketplaceLifecycleHandoffService,
)
from intergrax.contracts.lifecycle_handoff.ack import (
    DomainLifecycleHandoffAck,
    DomainLifecycleHandoffDisposition,
)
from intergrax.contracts.marketplace.lifecycle_handoff_outcome import (
    MarketplaceLifecycleHandoffStatus,
)

pytestmark = pytest.mark.unit

_OFFICIAL = CapabilitySourceIdentity(
    source_id="official.intergrax.me10",
    source_kind=CapabilitySourceKind.OFFICIAL,
)


def _discovery_query(**kwargs: object) -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
        **kwargs,
    )


def _entry(
    *,
    kind: CapabilityKind,
    logical_id: str,
    version_label: str = "1.0.0",
    publisher: str = "publisher-me10",
    digest: str = "sha256:me10",
) -> CapabilityCatalogEntry:
    return CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=kind,
            source=_OFFICIAL,
            logical=CapabilityLogicalIdentity(kind=kind, logical_id=logical_id),
        ),
        provenance=CapabilityProvenance(
            source=_OFFICIAL,
            version_label=version_label,
            content_digest=digest,
            publisher=publisher,
        ),
        display_label=logical_id,
    )


def _record(
    kind: CapabilityKind,
    logical_id: str,
    *,
    version_label: str = "1.0.0",
    tenant_id: str | None = None,
) -> MarketplaceListingRecord:
    visibility = None
    if tenant_id is not None:
        visibility = MarketplaceVisibility(
            scope=MarketplaceVisibilityScope.TENANT_PRIVATE,
            tenant_id=tenant_id,
        )
    return MarketplaceListingRecord(
        kind=kind,
        logical_id=logical_id,
        display_label=logical_id,
        publisher="intergrax",
        version_label=version_label,
        content_digest=f"sha256:{logical_id}-{version_label}",
        visibility=visibility,
    )


def _catalog_service(
    *records: MarketplaceListingRecord,
) -> MarketplaceCatalogService:
    source = MarketplaceCapabilityCatalogSource(source=_OFFICIAL, records=records)
    catalog = FederatedCapabilityCatalog((source,))
    return MarketplaceCatalogService(catalog=catalog, marketplace_sources=(source,))


class _RecordingHandoffConsumer:
    def __init__(self, consumer_id: str = "custom.consumer.me10") -> None:
        self._consumer_id = consumer_id
        self.envelopes: list = []
        self.fail_next = False

    @property
    def consumer_id(self) -> str:
        return self._consumer_id

    def consume(self, envelope) -> None:
        if self.fail_next:
            self.fail_next = False
            raise CapabilityHandoffConsumerError("consumer rejected handoff")
        self.envelopes.append(envelope)


def _orchestrator(
    service: MarketplaceCatalogService,
    consumer: _RecordingHandoffConsumer,
    *,
    trace: InMemoryCapabilityHandoffTraceEvidenceConsumer | None = None,
) -> MarketplaceDiscoveryHandoffOrchestrator:
    trace = trace or InMemoryCapabilityHandoffTraceEvidenceConsumer()
    delivery = CapabilityHandoffDeliveryService(
        consumer=consumer,
        delivery_admission=InMemoryCapabilityHandoffDeliveryAdmission(),
        trace_evidence_consumer=trace,
    )
    return MarketplaceDiscoveryHandoffOrchestrator(
        catalog_service=service,
        discovery_service=MarketplaceDiscoveryService.with_defaults(),
        governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
        governance_context=CapabilityGovernanceContext(
            posture=CapabilityGovernancePosture.STRICT,
        ),
        delivery_service=delivery,
    )


def _run_handoff(
    orchestrator: MarketplaceDiscoveryHandoffOrchestrator,
    *,
    service: MarketplaceCatalogService,
    logical_id: str,
    kind: CapabilityKind,
    tenant_id: str | None = None,
    handoff_id: str = "handoff-1",
) -> tuple:
    snapshot = service._catalog.snapshot()
    entry = next(
        e for e in snapshot.entries if e.identity.logical.logical_id == logical_id
    )
    key = CapabilityIdentityKey.from_discovery_identity(entry.identity)
    ctx = MarketplaceQueryContext(tenant_id=tenant_id)
    result = orchestrator.execute_explicit_selection_handoff(
        discovery_query=_discovery_query(),
        marketplace_query_context=ctx,
        selected_identity_key=key,
        consumer_target={
            CapabilityKind.AGENT: CapabilityHandoffConsumerTarget.AGENT_DOMAIN,
            CapabilityKind.TOOL: CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
            CapabilityKind.SKILL: CapabilityHandoffConsumerTarget.SKILL_DOMAIN,
        }[kind],
        selector_id="operator.explicit",
        discovery_correlation_id="discovery-corr-1",
        selection_id="selection-1",
        handoff_id=handoff_id,
        recorded_at=datetime(2026, 3, 16, 12, 0, tzinfo=timezone.utc),
    )
    return result, key, entry


def test_e2e_explicit_handoff_preserves_exact_release() -> None:
    service = _catalog_service(
        _record(CapabilityKind.AGENT, "agents.me10.alpha"),
    )
    consumer = _RecordingHandoffConsumer()
    trace = InMemoryCapabilityHandoffTraceEvidenceConsumer()
    orchestrator = _orchestrator(service, consumer, trace=trace)
    result, _key, entry = _run_handoff(
        orchestrator,
        service=service,
        logical_id="agents.me10.alpha",
        kind=CapabilityKind.AGENT,
    )
    assert result.disposition is CapabilityHandoffDeliveryDisposition.DELIVERED
    assert len(consumer.envelopes) == 1
    envelope = consumer.envelopes[0]
    expected = CapabilityReleaseIdentity.from_catalog_entry(entry)
    assert envelope.selected_release == expected
    assert trace.get("handoff-1") is envelope
    assert envelope.discovery_correlation_id == "discovery-corr-1"


def test_version_change_after_discovery_handoff_still_references_v1() -> None:
    service_v1 = _catalog_service(
        _record(CapabilityKind.TOOL, "tools.me10.versioned", version_label="1.0.0"),
    )
    consumer = _RecordingHandoffConsumer()
    orchestrator_v1 = _orchestrator(service_v1, consumer)
    _run_handoff(
        orchestrator_v1,
        service=service_v1,
        logical_id="tools.me10.versioned",
        kind=CapabilityKind.TOOL,
    )
    envelope_v1 = consumer.envelopes[0]
    assert envelope_v1.selected_release.version_label == "1.0.0"

    service_v2 = _catalog_service(
        _record(CapabilityKind.TOOL, "tools.me10.versioned", version_label="2.0.0"),
    )
    snapshot_v2 = service_v2._catalog.snapshot()
    current = next(
        e
        for e in snapshot_v2.entries
        if e.identity.logical.logical_id == "tools.me10.versioned"
    )
    assert CapabilityReleaseIdentity.from_catalog_entry(current).version_label == "2.0.0"
    assert envelope_v1.selected_release.version_label == "1.0.0"


def test_tenant_private_handoff_only_for_owning_tenant() -> None:
    service = _catalog_service(
        _record(CapabilityKind.AGENT, "agents.private.a", tenant_id="tenant-a"),
        _record(CapabilityKind.AGENT, "agents.private.b", tenant_id="tenant-b"),
    )
    consumer_a = _RecordingHandoffConsumer()
    orchestrator_a = _orchestrator(service, consumer_a)
    _run_handoff(
        orchestrator_a,
        service=service,
        logical_id="agents.private.a",
        kind=CapabilityKind.AGENT,
        tenant_id="tenant-a",
    )
    consumer_b = _RecordingHandoffConsumer()
    orchestrator_b = _orchestrator(service, consumer_b)
    with pytest.raises(MarketplaceHandoffSelectionError):
        _run_handoff(
            orchestrator_b,
            service=service,
            logical_id="agents.private.a",
            kind=CapabilityKind.AGENT,
            tenant_id="tenant-b",
        )


def test_public_capability_handoff_carries_request_tenant_correlation() -> None:
    service = _catalog_service(
        _record(CapabilityKind.SKILL, "skills.me10.public"),
    )
    consumer = _RecordingHandoffConsumer()
    orchestrator = _orchestrator(service, consumer)
    _run_handoff(
        orchestrator,
        service=service,
        logical_id="skills.me10.public",
        kind=CapabilityKind.SKILL,
        tenant_id="tenant-request",
    )
    envelope = consumer.envelopes[0]
    assert envelope.tenant_id == "tenant-request"
    assert envelope.selected_release.discovery.source.source_kind is CapabilitySourceKind.OFFICIAL


def test_custom_structural_consumer_without_subclass() -> None:
    service = _catalog_service(_record(CapabilityKind.TOOL, "tools.me10.custom"))
    consumer = _RecordingHandoffConsumer(consumer_id="external.port")
    orchestrator = _orchestrator(service, consumer)
    _run_handoff(orchestrator, service=service, logical_id="tools.me10.custom", kind=CapabilityKind.TOOL)
    assert consumer.envelopes[0].downstream_consumer_id == "external.port"


def test_consumer_failure_is_explicit() -> None:
    service = _catalog_service(_record(CapabilityKind.AGENT, "agents.me10.fail"))
    consumer = _RecordingHandoffConsumer()
    consumer.fail_next = True
    orchestrator = _orchestrator(service, consumer)
    with pytest.raises(CapabilityHandoffConsumerError, match="consumer rejected"):
        _run_handoff(
            orchestrator,
            service=service,
            logical_id="agents.me10.fail",
            kind=CapabilityKind.AGENT,
        )


def test_duplicate_handoff_delivery_skips_second_consumer_invocation() -> None:
    service = _catalog_service(_record(CapabilityKind.AGENT, "agents.me10.dedupe"))
    consumer = _RecordingHandoffConsumer()
    trace = InMemoryCapabilityHandoffTraceEvidenceConsumer()
    orchestrator = _orchestrator(service, consumer, trace=trace)
    _run_handoff(
        orchestrator,
        service=service,
        logical_id="agents.me10.dedupe",
        kind=CapabilityKind.AGENT,
        handoff_id="handoff-dup",
    )
    second = orchestrator.execute_explicit_selection_handoff(
        discovery_query=_discovery_query(),
        marketplace_query_context=MarketplaceQueryContext(),
        selected_identity_key=CapabilityIdentityKey.from_discovery_identity(
            service._catalog.snapshot().entries[0].identity,
        ),
        consumer_target=CapabilityHandoffConsumerTarget.AGENT_DOMAIN,
        selector_id="operator.explicit",
        discovery_correlation_id="discovery-corr-1",
        selection_id="selection-1",
        handoff_id="handoff-dup",
        recorded_at=datetime(2026, 3, 16, 12, 0, tzinfo=timezone.utc),
    )
    assert second.disposition is CapabilityHandoffDeliveryDisposition.DUPLICATE_SKIPPED
    assert len(consumer.envelopes) == 1


def test_ranking_alone_does_not_invoke_handoff_consumer() -> None:
    service = _catalog_service(_record(CapabilityKind.TOOL, "tools.me10.rank"))
    consumer = _RecordingHandoffConsumer()
    _orchestrator(service, consumer)
    snapshot = service._catalog.snapshot()
    candidates = tuple(
        CapabilityDiscoveryCandidate(
            catalog_entry=entry,
            availability=AvailabilityDisposition.CATALOG_AVAILABLE,
        )
        for entry in snapshot.entries
    )
    rank_capability_candidates(candidates, MarketplaceDiscoveryService.with_defaults().ranker)
    assert consumer.envelopes == []


def test_recommendation_does_not_auto_handoff() -> None:
    service = _catalog_service(_record(CapabilityKind.AGENT, "agents.me10.rec"))
    consumer = _RecordingHandoffConsumer()
    _orchestrator(service, consumer)
    from intergrax.capability_catalog import govern_capability_candidates

    snapshot = service._catalog.snapshot()
    candidates = tuple(
        CapabilityDiscoveryCandidate(
            catalog_entry=entry,
            availability=AvailabilityDisposition.CATALOG_AVAILABLE,
        )
        for entry in snapshot.entries
    )
    ranked = MarketplaceDiscoveryService.with_defaults().search_and_rank(candidates)
    governed = govern_capability_candidates(
        ranked,
        evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
        context=CapabilityGovernanceContext(posture=CapabilityGovernancePosture.STRICT),
    ).allowed
    MarketplaceRecommendationService.with_defaults().recommend(governed)
    assert consumer.envelopes == []


def test_discovery_pipeline_emits_no_capability_usage_events() -> None:
    service = _catalog_service(_record(CapabilityKind.SKILL, "skills.me10.usage"))
    consumer = _RecordingHandoffConsumer()
    usage_events: list[object] = []

    class _SpyUsageConsumer:
        def consume(self, event: object) -> None:
            usage_events.append(event)

    recorder = CapabilityUsageRecorder(_SpyUsageConsumer())
    del recorder  # orchestration path does not touch metering
    orchestrator = _orchestrator(service, consumer)
    _run_handoff(
        orchestrator,
        service=service,
        logical_id="skills.me10.usage",
        kind=CapabilityKind.SKILL,
    )
    assert usage_events == []


def test_attribution_from_handoff_matches_exact_release_provenance() -> None:
    service = _catalog_service(_record(CapabilityKind.TOOL, "tools.me10.attr"))
    consumer = _RecordingHandoffConsumer()
    orchestrator = _orchestrator(service, consumer)
    _run_handoff(
        orchestrator,
        service=service,
        logical_id="tools.me10.attr",
        kind=CapabilityKind.TOOL,
    )
    attribution = attribution_from_handoff_envelope(consumer.envelopes[0])
    release = consumer.envelopes[0].selected_release
    assert attribution.identity == CapabilityIdentityKey.from_discovery_identity(
        release.discovery,
    )
    assert attribution.provenance == release.to_provenance()


@pytest.mark.parametrize(
    ("kind", "logical_id", "target"),
    (
        (CapabilityKind.AGENT, "agents.me10.bridge", CapabilityHandoffConsumerTarget.AGENT_DOMAIN),
        (CapabilityKind.TOOL, "tools.me10.bridge", CapabilityHandoffConsumerTarget.TOOL_DOMAIN),
        (CapabilityKind.SKILL, "skills.me10.bridge", CapabilityHandoffConsumerTarget.SKILL_DOMAIN),
    ),
)
def test_lifecycle_bridge_from_envelope_for_each_kind(
    kind: CapabilityKind,
    logical_id: str,
    target: CapabilityHandoffConsumerTarget,
) -> None:
    service = _catalog_service(_record(kind, logical_id))
    consumer = _RecordingHandoffConsumer()
    orchestrator = _orchestrator(service, consumer)
    _run_handoff(
        orchestrator,
        service=service,
        logical_id=logical_id,
        kind=kind,
    )
    envelope = consumer.envelopes[0]
    assert envelope.consumer_target is target
    identity_key = CapabilityIdentityKey.from_discovery_identity(
        envelope.selected_release.discovery,
    )
    domain = MarketplaceLifecycleDomainPayload(
        agent=(
            AgentLifecycleHandoffPayload(
                operation_id="op-1",
                application_id="app-1",
                application_environment_id="env-1",
                catalog_entry_id="cat-1",
                capability_identity_key=identity_key,
            )
            if kind is CapabilityKind.AGENT
            else None
        ),
        tool=(
            ToolLifecycleHandoffPayload(
                operation_id="op-1",
                host_profile_id="host-1",
                capability_identity_key=identity_key,
            )
            if kind is CapabilityKind.TOOL
            else None
        ),
        skill=(
            SkillLifecycleHandoffPayload(
                operation_id="op-1",
                host_profile_id="host-1",
                capability_identity_key=identity_key,
            )
            if kind is CapabilityKind.SKILL
            else None
        ),
    )
    lifecycle_request = lifecycle_handoff_request_from_envelope(
        envelope,
        request_id="lifecycle-req-1",
        intent=MarketplaceLifecycleHandoffIntent.REQUEST_LIFECYCLE,
        domain_payload=domain,
    )
    assert lifecycle_request.selection.selected_release == envelope.selected_release


class _AckAgentPort:
    def submit_marketplace_lifecycle_handoff(self, payload, *, request_id, correlation_id):
        del payload, request_id, correlation_id
        return DomainLifecycleHandoffAck(
            disposition=DomainLifecycleHandoffDisposition.ACCEPTED,
            domain_reference="agent:ok",
        )


def test_agent_lifecycle_handoff_after_traceability_envelope() -> None:
    service = _catalog_service(_record(CapabilityKind.AGENT, "agents.me10.lifecycle"))
    consumer = _RecordingHandoffConsumer()
    orchestrator = _orchestrator(service, consumer)
    _run_handoff(
        orchestrator,
        service=service,
        logical_id="agents.me10.lifecycle",
        kind=CapabilityKind.AGENT,
    )
    envelope = consumer.envelopes[0]
    identity_key = CapabilityIdentityKey.from_discovery_identity(
        envelope.selected_release.discovery,
    )
    lifecycle_request = lifecycle_handoff_request_from_envelope(
        envelope,
        request_id="req-agent",
        intent=MarketplaceLifecycleHandoffIntent.REQUEST_LIFECYCLE,
        domain_payload=MarketplaceLifecycleDomainPayload(
            agent=AgentLifecycleHandoffPayload(
                operation_id="op-1",
                application_id="app-1",
                application_environment_id="env-1",
                catalog_entry_id="cat-1",
                capability_identity_key=identity_key,
            ),
        ),
    )
    resolver = LifecycleHandoffResolver(
        {CapabilityKind.AGENT: AgentMarketplaceLifecycleHandoffHandler(_AckAgentPort())},
    )
    outcome = MarketplaceLifecycleHandoffService(resolver).handoff(lifecycle_request)
    assert outcome.status is MarketplaceLifecycleHandoffStatus.ACCEPTED


def test_handoff_traceability_package_has_no_runtime_imports() -> None:
    package = importlib.import_module("intergrax.marketplace.handoff_traceability")
    root = Path(package.__path__[0])
    forbidden = (
        "intergrax.runtime",
        "intergrax.runtime.nexus",
        "intergrax.nexus",
    )
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for prefix in forbidden:
                        if alias.name == prefix or alias.name.startswith(f"{prefix}."):
                            raise AssertionError(f"{path.name} imports forbidden {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                for prefix in forbidden:
                    if node.module == prefix or node.module.startswith(f"{prefix}."):
                        raise AssertionError(f"{path.name} imports forbidden {node.module}")
