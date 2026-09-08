# © Artur Czarnecki. All rights reserved.

"""CAPABILITY-CATALOG-1 Stage 13 metering substrate tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.capability_catalog import (
    CapabilityCatalogEntry,
    CapabilityDiscoveryCandidate,
)
from intergrax.capability_metering import (
    CapabilityUsageAttribution,
    CapabilityUsageConflictError,
    CapabilityUsageRecorder,
    InMemoryCapabilityUsageConsumer,
    attribution_from_discovery_candidate,
    project_capability_usage_summary,
)
from intergrax.contracts.capability_catalog import (
    AvailabilityDisposition,
    CapabilityDiscoveryIdentity,
    CapabilityIdentityKey,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityProvenance,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.capability_metering import (
    CapabilityUsageKind,
    CapabilityUsageOutcome,
    build_capability_usage_event,
)
from intergrax.contracts.execution_identity import mint_event_id

pytestmark = pytest.mark.unit


def _source(
    *,
    source_id: str,
    source_kind: CapabilitySourceKind,
) -> CapabilitySourceIdentity:
    return CapabilitySourceIdentity(source_id=source_id, source_kind=source_kind)


def _entry(
    *,
    kind: CapabilityKind,
    source_id: str,
    source_kind: CapabilitySourceKind,
    logical_id: str,
    version_label: str | None = None,
    content_digest: str | None = None,
    publisher: str | None = None,
) -> CapabilityCatalogEntry:
    source = _source(source_id=source_id, source_kind=source_kind)
    return CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=kind,
            source=source,
            logical=CapabilityLogicalIdentity(kind=kind, logical_id=logical_id),
        ),
        provenance=CapabilityProvenance(
            source=source,
            version_label=version_label,
            content_digest=content_digest,
            publisher=publisher,
        ),
    )


def _candidate(entry: CapabilityCatalogEntry) -> CapabilityDiscoveryCandidate:
    return CapabilityDiscoveryCandidate(
        catalog_entry=entry,
        availability=AvailabilityDisposition.HOST_AVAILABLE,
    )


def test_attribution_round_trip_from_discovery_candidate() -> None:
    entry = _entry(
        kind=CapabilityKind.TOOL,
        source_id="marketplace.x",
        source_kind=CapabilitySourceKind.OFFICIAL,
        logical_id="tool.search",
        version_label="2.1",
        content_digest="sha256:abc",
        publisher="vendor-x",
    )
    candidate = _candidate(entry)
    attribution = attribution_from_discovery_candidate(candidate)
    consumer = InMemoryCapabilityUsageConsumer()
    recorder = CapabilityUsageRecorder(consumer=consumer)
    event = recorder.record(
        tenant_id="tenant-a",
        attribution=attribution,
        usage_kind=CapabilityUsageKind.EXECUTION,
        outcome=CapabilityUsageOutcome.SUCCEEDED,
    )
    report = project_capability_usage_summary(consumer, tenant_id="tenant-a")
    assert event.identity.logical_id == "tool.search"
    assert event.identity.source_id == "marketplace.x"
    assert event.provenance.publisher == "vendor-x"
    assert len(report.identity_rollups) == 1
    assert report.identity_rollups[0].identity == event.identity
    assert len(report.publisher_rollups) == 1
    assert report.publisher_rollups[0].publisher == "vendor-x"
    assert report.publisher_rollups[0].total_quantity == 1


def test_agent_usage_event_preserves_enterprise_private_source() -> None:
    entry = _entry(
        kind=CapabilityKind.AGENT,
        source_id="enterprise.private",
        source_kind=CapabilitySourceKind.ENTERPRISE_PRIVATE,
        logical_id="agents.researcher",
        publisher=None,
    )
    attribution = attribution_from_discovery_candidate(_candidate(entry))
    event = build_capability_usage_event(
        tenant_id="tenant-a",
        identity=attribution.identity,
        provenance=attribution.provenance,
        usage_kind=CapabilityUsageKind.DELEGATION,
        outcome=CapabilityUsageOutcome.SUCCEEDED,
    )
    assert event.identity.kind is CapabilityKind.AGENT
    assert event.identity.source_kind is CapabilitySourceKind.ENTERPRISE_PRIVATE
    assert event.provenance.publisher is None


def test_same_logical_id_different_sources_remain_distinct() -> None:
    official = attribution_from_discovery_candidate(
        _candidate(
            _entry(
                kind=CapabilityKind.TOOL,
                source_id="official.catalog",
                source_kind=CapabilitySourceKind.OFFICIAL,
                logical_id="tool.search",
            ),
        ),
    )
    private = attribution_from_discovery_candidate(
        _candidate(
            _entry(
                kind=CapabilityKind.TOOL,
                source_id="enterprise.private",
                source_kind=CapabilitySourceKind.ENTERPRISE_PRIVATE,
                logical_id="tool.search",
            ),
        ),
    )
    consumer = InMemoryCapabilityUsageConsumer()
    recorder = CapabilityUsageRecorder(consumer=consumer)
    recorder.record(
        tenant_id="tenant-a",
        attribution=official,
        usage_kind=CapabilityUsageKind.EXECUTION,
        outcome=CapabilityUsageOutcome.SUCCEEDED,
    )
    recorder.record(
        tenant_id="tenant-a",
        attribution=private,
        usage_kind=CapabilityUsageKind.EXECUTION,
        outcome=CapabilityUsageOutcome.SUCCEEDED,
    )
    report = project_capability_usage_summary(consumer, tenant_id="tenant-a")
    assert len(report.identity_rollups) == 2
    assert report.identity_rollups[0].identity != report.identity_rollups[1].identity


def test_historical_provenance_snapshots_are_preserved() -> None:
    consumer = InMemoryCapabilityUsageConsumer()
    recorder = CapabilityUsageRecorder(consumer=consumer)
    for version_label, digest, publisher in (
        ("1.0.0", "sha256:v1", "vendor-v1"),
        ("2.0.0", "sha256:v2", "vendor-v2"),
    ):
        entry = _entry(
            kind=CapabilityKind.TOOL,
            source_id="marketplace.x",
            source_kind=CapabilitySourceKind.OFFICIAL,
            logical_id="tool.search",
            version_label=version_label,
            content_digest=digest,
            publisher=publisher,
        )
        recorder.record(
            tenant_id="tenant-a",
            attribution=attribution_from_discovery_candidate(_candidate(entry)),
            usage_kind=CapabilityUsageKind.EXECUTION,
            outcome=CapabilityUsageOutcome.SUCCEEDED,
        )
    events = consumer.snapshot()
    assert events[0].provenance.version_label == "1.0.0"
    assert events[1].provenance.version_label == "2.0.0"
    assert events[0].provenance.publisher == "vendor-v1"
    assert events[1].provenance.publisher == "vendor-v2"


def _tool_identity() -> CapabilityIdentityKey:
    return CapabilityIdentityKey(
        kind=CapabilityKind.TOOL,
        source_id="marketplace.x",
        source_kind=CapabilitySourceKind.OFFICIAL,
        logical_id="tool.search",
    )


def _tool_provenance() -> CapabilityProvenance:
    return CapabilityProvenance(
        source=_source(
            source_id="marketplace.x",
            source_kind=CapabilitySourceKind.OFFICIAL,
        ),
        version_label="2.1",
        publisher="vendor-x",
    )


def test_consumer_is_idempotent_for_exact_duplicate_event() -> None:
    event = build_capability_usage_event(
        tenant_id="tenant-a",
        identity=_tool_identity(),
        provenance=_tool_provenance(),
        usage_kind=CapabilityUsageKind.EXECUTION,
        outcome=CapabilityUsageOutcome.SUCCEEDED,
        quantity=5,
        event_id=mint_event_id(),
    )
    consumer = InMemoryCapabilityUsageConsumer()
    consumer.consume(event)
    consumer.consume(event)
    report = project_capability_usage_summary(consumer, tenant_id="tenant-a")
    assert report.identity_rollups[0].total_quantity == 5


def test_consumer_fails_closed_on_duplicate_event_id_conflict() -> None:
    event_id = mint_event_id()
    first = build_capability_usage_event(
        tenant_id="tenant-a",
        identity=_tool_identity(),
        provenance=_tool_provenance(),
        usage_kind=CapabilityUsageKind.EXECUTION,
        outcome=CapabilityUsageOutcome.SUCCEEDED,
        event_id=event_id,
    )
    second = build_capability_usage_event(
        tenant_id="tenant-a",
        identity=_tool_identity(),
        provenance=_tool_provenance(),
        usage_kind=CapabilityUsageKind.EXECUTION,
        outcome=CapabilityUsageOutcome.FAILED,
        event_id=event_id,
    )
    consumer = InMemoryCapabilityUsageConsumer()
    consumer.consume(first)
    with pytest.raises(CapabilityUsageConflictError):
        consumer.consume(second)


def test_tenant_separation_in_summary_projection() -> None:
    attribution = attribution_from_discovery_candidate(
        _candidate(
            _entry(
                kind=CapabilityKind.TOOL,
                source_id="marketplace.x",
                source_kind=CapabilitySourceKind.OFFICIAL,
                logical_id="tool.search",
            ),
        ),
    )
    consumer = InMemoryCapabilityUsageConsumer()
    recorder = CapabilityUsageRecorder(consumer=consumer)
    recorder.record(
        tenant_id="tenant-a",
        attribution=attribution,
        usage_kind=CapabilityUsageKind.EXECUTION,
        outcome=CapabilityUsageOutcome.SUCCEEDED,
    )
    recorder.record(
        tenant_id="tenant-b",
        attribution=attribution,
        usage_kind=CapabilityUsageKind.EXECUTION,
        outcome=CapabilityUsageOutcome.SUCCEEDED,
    )
    report_a = project_capability_usage_summary(consumer, tenant_id="tenant-a")
    report_b = project_capability_usage_summary(consumer, tenant_id="tenant-b")
    assert report_a.identity_rollups[0].total_quantity == 1
    assert report_b.identity_rollups[0].total_quantity == 1
    assert len(project_capability_usage_summary(consumer, tenant_id="tenant-a").identity_rollups) == 1


def test_recorder_without_consumer_has_no_side_effects() -> None:
    attribution = attribution_from_discovery_candidate(
        _candidate(
            _entry(
                kind=CapabilityKind.TOOL,
                source_id="marketplace.x",
                source_kind=CapabilitySourceKind.OFFICIAL,
                logical_id="tool.search",
            ),
        ),
    )
    recorder = CapabilityUsageRecorder()
    event = recorder.record(
        tenant_id="tenant-a",
        attribution=attribution,
        usage_kind=CapabilityUsageKind.EXECUTION,
        outcome=CapabilityUsageOutcome.SUCCEEDED,
    )
    assert event.tenant_id == "tenant-a"


def test_attribution_rejects_identity_provenance_mismatch() -> None:
    with pytest.raises(ValidationError, match="source_id"):
        CapabilityUsageAttribution(
            identity=CapabilityIdentityKey(
                kind=CapabilityKind.TOOL,
                source_id="official.catalog",
                source_kind=CapabilitySourceKind.OFFICIAL,
                logical_id="tool.search",
            ),
            provenance=CapabilityProvenance(
                source=_source(
                    source_id="enterprise.private",
                    source_kind=CapabilitySourceKind.ENTERPRISE_PRIVATE,
                ),
            ),
        )


def test_identity_rollup_sums_quantity_by_outcome() -> None:
    consumer = InMemoryCapabilityUsageConsumer()
    consumer.consume(
        build_capability_usage_event(
            tenant_id="tenant-a",
            identity=_tool_identity(),
            provenance=_tool_provenance(),
            usage_kind=CapabilityUsageKind.EXECUTION,
            outcome=CapabilityUsageOutcome.SUCCEEDED,
            quantity=5,
        ),
    )
    consumer.consume(
        build_capability_usage_event(
            tenant_id="tenant-a",
            identity=_tool_identity(),
            provenance=_tool_provenance(),
            usage_kind=CapabilityUsageKind.EXECUTION,
            outcome=CapabilityUsageOutcome.FAILED,
            quantity=2,
        ),
    )
    rollup = project_capability_usage_summary(consumer, tenant_id="tenant-a").identity_rollups[0]
    assert rollup.total_quantity == 7
    assert rollup.succeeded_quantity == 5
    assert rollup.failed_quantity == 2
    assert rollup.cancelled_quantity == 0
    assert rollup.timeout_quantity == 0


def test_publisher_rollup_sums_quantity() -> None:
    consumer = InMemoryCapabilityUsageConsumer()
    for quantity in (3, 4):
        consumer.consume(
            build_capability_usage_event(
                tenant_id="tenant-a",
                identity=_tool_identity(),
                provenance=_tool_provenance(),
                usage_kind=CapabilityUsageKind.EXECUTION,
                outcome=CapabilityUsageOutcome.SUCCEEDED,
                quantity=quantity,
            ),
        )
    report = project_capability_usage_summary(consumer, tenant_id="tenant-a")
    assert report.publisher_rollups[0].publisher == "vendor-x"
    assert report.publisher_rollups[0].total_quantity == 7


def test_consumer_fails_closed_on_duplicate_event_id_quantity_conflict() -> None:
    event_id = mint_event_id()
    first = build_capability_usage_event(
        tenant_id="tenant-a",
        identity=_tool_identity(),
        provenance=_tool_provenance(),
        usage_kind=CapabilityUsageKind.EXECUTION,
        outcome=CapabilityUsageOutcome.SUCCEEDED,
        quantity=5,
        event_id=event_id,
    )
    second = build_capability_usage_event(
        tenant_id="tenant-a",
        identity=_tool_identity(),
        provenance=_tool_provenance(),
        usage_kind=CapabilityUsageKind.EXECUTION,
        outcome=CapabilityUsageOutcome.SUCCEEDED,
        quantity=6,
        event_id=event_id,
    )
    consumer = InMemoryCapabilityUsageConsumer()
    consumer.consume(first)
    with pytest.raises(CapabilityUsageConflictError):
        consumer.consume(second)


def test_tenant_quantity_separation_in_summary_projection() -> None:
    attribution = attribution_from_discovery_candidate(
        _candidate(
            _entry(
                kind=CapabilityKind.TOOL,
                source_id="marketplace.x",
                source_kind=CapabilitySourceKind.OFFICIAL,
                logical_id="tool.search",
            ),
        ),
    )
    consumer = InMemoryCapabilityUsageConsumer()
    recorder = CapabilityUsageRecorder(consumer=consumer)
    recorder.record(
        tenant_id="tenant-a",
        attribution=attribution,
        usage_kind=CapabilityUsageKind.EXECUTION,
        outcome=CapabilityUsageOutcome.SUCCEEDED,
        quantity=5,
    )
    recorder.record(
        tenant_id="tenant-b",
        attribution=attribution,
        usage_kind=CapabilityUsageKind.EXECUTION,
        outcome=CapabilityUsageOutcome.SUCCEEDED,
        quantity=7,
    )
    report_a = project_capability_usage_summary(consumer, tenant_id="tenant-a")
    report_b = project_capability_usage_summary(consumer, tenant_id="tenant-b")
    assert report_a.identity_rollups[0].total_quantity == 5
    assert report_b.identity_rollups[0].total_quantity == 7


def test_same_logical_id_different_sources_sum_quantity_separately() -> None:
    official = attribution_from_discovery_candidate(
        _candidate(
            _entry(
                kind=CapabilityKind.TOOL,
                source_id="official.catalog",
                source_kind=CapabilitySourceKind.OFFICIAL,
                logical_id="tool.search",
            ),
        ),
    )
    private = attribution_from_discovery_candidate(
        _candidate(
            _entry(
                kind=CapabilityKind.TOOL,
                source_id="enterprise.private",
                source_kind=CapabilitySourceKind.ENTERPRISE_PRIVATE,
                logical_id="tool.search",
            ),
        ),
    )
    consumer = InMemoryCapabilityUsageConsumer()
    recorder = CapabilityUsageRecorder(consumer=consumer)
    recorder.record(
        tenant_id="tenant-a",
        attribution=official,
        usage_kind=CapabilityUsageKind.EXECUTION,
        outcome=CapabilityUsageOutcome.SUCCEEDED,
        quantity=3,
    )
    recorder.record(
        tenant_id="tenant-a",
        attribution=private,
        usage_kind=CapabilityUsageKind.EXECUTION,
        outcome=CapabilityUsageOutcome.SUCCEEDED,
        quantity=4,
    )
    report = project_capability_usage_summary(consumer, tenant_id="tenant-a")
    quantities = {
        rollup.identity.source_kind: rollup.total_quantity for rollup in report.identity_rollups
    }
    assert quantities[CapabilitySourceKind.OFFICIAL] == 3
    assert quantities[CapabilitySourceKind.ENTERPRISE_PRIVATE] == 4
