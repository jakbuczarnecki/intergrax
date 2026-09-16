# © Artur Czarnecki. All rights reserved.

"""ME-8 — commercial presentation vs usage metering boundary qualification."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from intergrax.capability_catalog import (
    AvailabilityPreservingGovernanceEvaluator,
    CapabilityCatalogEntry,
    CapabilityDiscoveryCandidate,
    StableIdentityRanker,
    govern_capability_candidates,
    rank_capability_candidates,
)
from intergrax.contracts.capability_catalog.governance import GovernanceDisposition
from intergrax.capability_metering import (
    CapabilityUsageAttribution,
    CapabilityUsageRecorder,
    attribution_from_discovery_candidate,
)
from intergrax.capability_metering.errors import CapabilityMeteringError
from intergrax.contracts.capability_catalog import (
    AvailabilityDisposition,
    CapabilityDiscoveryIdentity,
    CapabilityGovernanceContext,
    CapabilityIdentityKey,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityProvenance,
    CapabilityRankingContext,
    CapabilityReleaseIdentity,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.capability_metering import (
    CapabilityUsageEvent,
    CapabilityUsageKind,
    CapabilityUsageOutcome,
    build_capability_usage_event,
)
from intergrax.contracts.marketplace.commercial import (
    CommercialModel,
    MarketplaceCommercialMetadata,
)

pytestmark = pytest.mark.unit

_GOVERNANCE_MODULES = (
    "intergrax.capability_catalog.governance",
    "intergrax.capability_catalog.governance_validation",
    "intergrax.capability_catalog.adapters.agent_governance",
    "intergrax.capability_catalog.adapters.tool_governance",
    "intergrax.capability_catalog.adapters.skill_governance",
)

_COMMERCIAL_CONTRACT = "intergrax.contracts.marketplace.commercial"


def _source() -> CapabilitySourceIdentity:
    return CapabilitySourceIdentity(
        source_id="official.me8",
        source_kind=CapabilitySourceKind.OFFICIAL,
    )


def _entry(
    *,
    kind: CapabilityKind,
    logical_id: str,
    version_label: str = "3.2.1",
    publisher: str = "publisher-me8",
) -> CapabilityCatalogEntry:
    source = _source()
    return CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=kind,
            source=source,
            logical=CapabilityLogicalIdentity(kind=kind, logical_id=logical_id),
        ),
        provenance=CapabilityProvenance(
            source=source,
            version_label=version_label,
            content_digest="sha256:me8-release",
            publisher=publisher,
        ),
    )


def _candidate(entry: CapabilityCatalogEntry) -> CapabilityDiscoveryCandidate:
    return CapabilityDiscoveryCandidate(
        catalog_entry=entry,
        availability=AvailabilityDisposition.CATALOG_AVAILABLE,
    )


def test_usage_event_is_pricing_agnostic() -> None:
    monetary_fields = frozenset(
        {
            "price",
            "cost",
            "billing",
            "final_price",
            "amount_due",
            "invoice_id",
            "tax",
            "discount_applied",
            "settlement_status",
            "payment_status",
            "minor_units",
            "currency_code",
        },
    )
    assert not monetary_fields.intersection(CapabilityUsageEvent.model_fields)
    event = build_capability_usage_event(
        tenant_id="tenant-me8",
        identity=CapabilityIdentityKey(
            kind=CapabilityKind.TOOL,
            source_id="official.me8",
            source_kind=CapabilitySourceKind.OFFICIAL,
            logical_id="tools.me8",
        ),
        provenance=_entry(kind=CapabilityKind.TOOL, logical_id="tools.me8").provenance,
        usage_kind=CapabilityUsageKind.EXECUTION,
        outcome=CapabilityUsageOutcome.SUCCEEDED,
    )
    payload = event.model_dump(mode="json")
    assert not monetary_fields.intersection(payload)


def test_usage_event_preserves_capability_release_attribution() -> None:
    entry = _entry(kind=CapabilityKind.AGENT, logical_id="agents.me8")
    release = CapabilityReleaseIdentity.from_catalog_entry(entry)
    attribution = attribution_from_discovery_candidate(_candidate(entry))
    event = build_capability_usage_event(
        tenant_id="tenant-me8",
        identity=attribution.identity,
        provenance=attribution.provenance,
        usage_kind=CapabilityUsageKind.DELEGATION,
        outcome=CapabilityUsageOutcome.SUCCEEDED,
        quantity=2,
    )
    assert event.provenance.version_label == release.version_label
    assert event.provenance.content_digest == release.content_digest
    assert event.provenance.publisher == release.publisher
    assert event.identity.logical_id == release.discovery.logical.logical_id


class _CustomUsageSink:
    """External metering adapter — not a subclass of platform consumer types."""

    def __init__(self) -> None:
        self.events: list[CapabilityUsageEvent] = []

    def consume(self, event: CapabilityUsageEvent) -> None:
        self.events.append(event)


def test_custom_usage_sink_plugs_in_without_core_changes() -> None:
    sink = _CustomUsageSink()
    recorder = CapabilityUsageRecorder(consumer=sink)
    entry = _entry(kind=CapabilityKind.TOOL, logical_id="tools.custom-sink")
    event = recorder.record(
        tenant_id="tenant-me8",
        attribution=attribution_from_discovery_candidate(_candidate(entry)),
        usage_kind=CapabilityUsageKind.EXECUTION,
        outcome=CapabilityUsageOutcome.SUCCEEDED,
    )
    assert len(sink.events) == 1
    assert sink.events[0] == event


class _FailingUsageSink:
    def consume(self, event: CapabilityUsageEvent) -> None:
        raise CapabilityMeteringError("sink unavailable")


def test_usage_sink_failure_is_typed_and_not_silently_dropped() -> None:
    recorder = CapabilityUsageRecorder(consumer=_FailingUsageSink())
    entry = _entry(kind=CapabilityKind.TOOL, logical_id="tools.sink-fail")
    with pytest.raises(CapabilityMeteringError, match="sink unavailable"):
        recorder.record(
            tenant_id="tenant-me8",
            attribution=attribution_from_discovery_candidate(_candidate(entry)),
            usage_kind=CapabilityUsageKind.EXECUTION,
            outcome=CapabilityUsageOutcome.SUCCEEDED,
        )


def test_marketplace_commercial_metadata_does_not_affect_ranking() -> None:
    free_meta = MarketplaceCommercialMetadata(commercial_model=CommercialModel.FREE)
    paid_meta = MarketplaceCommercialMetadata(
        commercial_model=CommercialModel.PAID,
        display_price="99.00 USD / month",
        pricing_reference="plan:enterprise-gold",
    )
    assert free_meta.commercial_model is not paid_meta.commercial_model

    alpha = _candidate(_entry(kind=CapabilityKind.TOOL, logical_id="tools.rank.alpha"))
    beta = _candidate(_entry(kind=CapabilityKind.TOOL, logical_id="tools.rank.beta"))
    ranker = StableIdentityRanker()
    context = CapabilityRankingContext()
    ranked_alpha_first = rank_capability_candidates((alpha, beta), ranker, context=context)
    ranked_beta_first = rank_capability_candidates((beta, alpha), ranker, context=context)
    assert [item.candidate.identity.logical.logical_id for item in ranked_alpha_first] == [
        "tools.rank.alpha",
        "tools.rank.beta",
    ]
    assert [item.candidate.identity.logical.logical_id for item in ranked_beta_first] == [
        "tools.rank.alpha",
        "tools.rank.beta",
    ]


def test_marketplace_commercial_metadata_does_not_affect_governance() -> None:
    for module_name in _GOVERNANCE_MODULES:
        module = importlib.import_module(module_name)
        path = Path(module.__file__)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported = node.module
                if imported == _COMMERCIAL_CONTRACT or imported.startswith(
                    f"{_COMMERCIAL_CONTRACT}.",
                ):
                    raise AssertionError(f"{module_name} imports commercial presentation contract")

    entry = _entry(kind=CapabilityKind.AGENT, logical_id="agents.gov")
    ranked = rank_capability_candidates(
        (_candidate(entry),),
        StableIdentityRanker(),
        context=CapabilityRankingContext(),
    )
    governed = govern_capability_candidates(
        tuple(ranked),
        evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
        context=CapabilityGovernanceContext(),
    )
    assert len(governed.allowed) == 1
    assert governed.allowed[0].evidence[-1].disposition is GovernanceDisposition.ALLOWED


def test_mixed_agent_tool_skill_usage_uses_common_contract() -> None:
    agent_entry = _entry(kind=CapabilityKind.AGENT, logical_id="agents.mixed")
    tool_entry = _entry(kind=CapabilityKind.TOOL, logical_id="tools.mixed")
    recorder = CapabilityUsageRecorder()
    agent_event = recorder.record(
        tenant_id="tenant-me8",
        attribution=attribution_from_discovery_candidate(_candidate(agent_entry)),
        usage_kind=CapabilityUsageKind.DELEGATION,
        outcome=CapabilityUsageOutcome.SUCCEEDED,
    )
    tool_event = recorder.record(
        tenant_id="tenant-me8",
        attribution=attribution_from_discovery_candidate(_candidate(tool_entry)),
        usage_kind=CapabilityUsageKind.EXECUTION,
        outcome=CapabilityUsageOutcome.SUCCEEDED,
    )
    assert type(agent_event) is CapabilityUsageEvent
    assert type(tool_event) is CapabilityUsageEvent
    assert agent_event.schema_version == tool_event.schema_version

    for kind in (CapabilityKind.AGENT, CapabilityKind.TOOL, CapabilityKind.SKILL):
        meta = MarketplaceCommercialMetadata(
            commercial_model=CommercialModel.PAID,
            display_price=f"display-{kind.value}",
            pricing_reference=f"ref:{kind.value}",
        )
        assert meta.pricing_reference == f"ref:{kind.value}"


def test_commercial_metadata_is_presentation_only() -> None:
    meta = MarketplaceCommercialMetadata(
        commercial_model=CommercialModel.PAID,
        display_price="from $10",
        pricing_reference="external:stripe-price_abc",
    )
    dumped = meta.model_dump(mode="json")
    assert "amount_due" not in dumped
    assert "invoice_id" not in dumped
    assert dumped["display_price"] == "from $10"


def test_usage_event_idempotency_identity_is_stable() -> None:
    from intergrax.contracts.execution_identity import mint_event_id

    event_id = mint_event_id()
    first = build_capability_usage_event(
        tenant_id="tenant-me8",
        identity=CapabilityIdentityKey(
            kind=CapabilityKind.TOOL,
            source_id="official.me8",
            source_kind=CapabilitySourceKind.OFFICIAL,
            logical_id="tools.idem",
        ),
        provenance=_entry(kind=CapabilityKind.TOOL, logical_id="tools.idem").provenance,
        usage_kind=CapabilityUsageKind.EXECUTION,
        outcome=CapabilityUsageOutcome.SUCCEEDED,
        event_id=event_id,
    )
    second = build_capability_usage_event(
        tenant_id="tenant-me8",
        identity=first.identity,
        provenance=first.provenance,
        usage_kind=first.usage_kind,
        outcome=first.outcome,
        quantity=first.quantity,
        event_id=event_id,
    )
    assert first == second
    assert first.event_id == second.event_id


def test_duplicate_usage_delivery_can_be_detected_by_event_identity() -> None:
    from intergrax.capability_metering import InMemoryCapabilityUsageConsumer
    from intergrax.capability_metering.errors import CapabilityUsageConflictError
    from intergrax.contracts.execution_identity import mint_event_id

    event_id = mint_event_id()
    attribution = CapabilityUsageAttribution(
        identity=CapabilityIdentityKey(
            kind=CapabilityKind.TOOL,
            source_id="official.me8",
            source_kind=CapabilitySourceKind.OFFICIAL,
            logical_id="tools.dup",
        ),
        provenance=_entry(kind=CapabilityKind.TOOL, logical_id="tools.dup").provenance,
    )
    first = build_capability_usage_event(
        tenant_id="tenant-me8",
        identity=attribution.identity,
        provenance=attribution.provenance,
        usage_kind=CapabilityUsageKind.EXECUTION,
        outcome=CapabilityUsageOutcome.SUCCEEDED,
        event_id=event_id,
    )
    conflicting = build_capability_usage_event(
        tenant_id="tenant-me8",
        identity=attribution.identity,
        provenance=attribution.provenance,
        usage_kind=CapabilityUsageKind.EXECUTION,
        outcome=CapabilityUsageOutcome.FAILED,
        event_id=event_id,
    )
    consumer = InMemoryCapabilityUsageConsumer()
    consumer.consume(first)
    with pytest.raises(CapabilityUsageConflictError):
        consumer.consume(conflicting)
