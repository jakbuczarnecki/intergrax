# © Artur Czarnecki. All rights reserved.

"""S24-GAP-02-P1 tool qualification staging consumer tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from intergrax.contracts.capability_catalog import (
    CapabilityDiscoveryIdentity,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityReleaseIdentity,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.marketplace import (
    CapabilityDiscoveryTraceFacts,
    CapabilityHandoffConsumerError,
    CapabilityHandoffConsumerFailureDisposition,
    CapabilityHandoffEnvelope,
    CapabilityMarketplaceExplicitSelection,
    MarketplaceQueryContext,
)
from intergrax.contracts.marketplace.handoff_traceability import (
    CapabilityHandoffConsumerTarget,
)
from intergrax.contracts.tools.marketplace_qualified_capability import (
    MarketplaceQualifiedToolStage,
    MarketplaceQualifiedToolStageIntegrityError,
    MarketplaceQualifiedToolStageRepository,
    MarketplaceQualifiedToolStageUnavailableError,
    MarketplaceQualifiedToolStageWriteResult,
)
from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.marketplace.handoff.adapters.tool_qualification_staging_consumer import (
    TOOL_QUALIFICATION_STAGING_CONSUMER_ID,
    ToolQualificationStagingConsumer,
)
from intergrax.tools.marketplace_qualified_capability_staging import (
    DocumentStoreMarketplaceQualifiedToolStageRepository,
)

pytestmark = pytest.mark.unit

_SOURCE = CapabilitySourceIdentity(
    source_id="official.marketplace.gap02",
    source_kind=CapabilitySourceKind.OFFICIAL,
)


def _tool_release(version_label: str = "1.0.0") -> CapabilityReleaseIdentity:
    return CapabilityReleaseIdentity(
        discovery=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=_SOURCE,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id="tools.consumer",
            ),
        ),
        publisher="publisher",
        version_label=version_label,
        content_digest="sha256:consumer",
        package_reference="pkg://consumer",
    )


def _agent_release() -> CapabilityReleaseIdentity:
    return CapabilityReleaseIdentity(
        discovery=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.AGENT,
            source=_SOURCE,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.AGENT,
                logical_id="agents.consumer",
            ),
        ),
        publisher="publisher",
        version_label="1.0.0",
    )


def _trace(tenant_id: str = "tenant-1") -> CapabilityDiscoveryTraceFacts:
    return CapabilityDiscoveryTraceFacts(
        discovery_correlation_id="discovery-1",
        marketplace_query_context=MarketplaceQueryContext(tenant_id=tenant_id),
        visible_candidate_count=1,
        governed_admissible_count=1,
    )


def _selection(release: CapabilityReleaseIdentity) -> CapabilityMarketplaceExplicitSelection:
    return CapabilityMarketplaceExplicitSelection(
        selection_id="selection-1",
        discovery_correlation_id="discovery-1",
        selected_release=release,
        selector_id="selector-1",
    )


def _envelope(
    release: CapabilityReleaseIdentity | None = None,
    *,
    tenant_id: str | None = "tenant-1",
    consumer_target: CapabilityHandoffConsumerTarget = (
        CapabilityHandoffConsumerTarget.TOOL_DOMAIN
    ),
) -> CapabilityHandoffEnvelope:
    release = release or _tool_release()
    return CapabilityHandoffEnvelope(
        handoff_id="handoff-1",
        tenant_id=tenant_id,
        selected_release=release,
        discovery_correlation_id="discovery-1",
        selection_id="selection-1",
        consumer_target=consumer_target,
        downstream_consumer_id=TOOL_QUALIFICATION_STAGING_CONSUMER_ID,
        discovery_trace=_trace(tenant_id or "tenant-1"),
        explicit_selection=_selection(release),
        recorded_at=datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc),
    )


def _consumer() -> tuple[ToolQualificationStagingConsumer, MarketplaceQualifiedToolStageRepository]:
    store = InMemoryDocumentStore()
    repository = DocumentStoreMarketplaceQualifiedToolStageRepository(store)
    return ToolQualificationStagingConsumer(repository), repository


def test_tool_envelope_persists_stage() -> None:
    consumer, repo = _consumer()
    envelope = _envelope()
    consumer.consume(envelope)
    loaded = repo.get(tenant_id="tenant-1", handoff_id="handoff-1")
    assert loaded is not None
    assert loaded.handoff_id == envelope.handoff_id


def test_selected_release_copied_exactly() -> None:
    consumer, repo = _consumer()
    release = _tool_release("3.4.5")
    envelope = _envelope(release)
    consumer.consume(envelope)
    loaded = repo.get(tenant_id="tenant-1", handoff_id="handoff-1")
    assert loaded is not None
    assert loaded.selected_release == release


def test_tenant_correlation_selection_preserved() -> None:
    consumer, repo = _consumer()
    envelope = _envelope()
    consumer.consume(envelope)
    loaded = repo.get(tenant_id="tenant-1", handoff_id="handoff-1")
    assert loaded is not None
    assert loaded.tenant_id == "tenant-1"
    assert loaded.discovery_correlation_id == "discovery-1"
    assert loaded.selection_id == "selection-1"


def test_duplicate_identical_envelope_is_safe() -> None:
    consumer, repo = _consumer()
    envelope = _envelope()
    consumer.consume(envelope)
    consumer.consume(envelope)
    loaded = repo.get(tenant_id="tenant-1", handoff_id="handoff-1")
    assert loaded is not None


def test_conflicting_stage_is_blocked() -> None:
    consumer, _repo = _consumer()
    consumer.consume(_envelope(_tool_release("1.0.0")))
    with pytest.raises(CapabilityHandoffConsumerError) as exc_info:
        consumer.consume(_envelope(_tool_release("2.0.0")))
    assert exc_info.value.disposition is CapabilityHandoffConsumerFailureDisposition.BLOCKED


def test_missing_tenant_is_blocked() -> None:
    consumer, _repo = _consumer()
    release = _tool_release()
    envelope = CapabilityHandoffEnvelope(
        handoff_id="handoff-1",
        tenant_id=None,
        selected_release=release,
        discovery_correlation_id="discovery-1",
        selection_id="selection-1",
        consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
        downstream_consumer_id=TOOL_QUALIFICATION_STAGING_CONSUMER_ID,
        discovery_trace=CapabilityDiscoveryTraceFacts(
            discovery_correlation_id="discovery-1",
            marketplace_query_context=MarketplaceQueryContext(tenant_id=None),
            visible_candidate_count=1,
            governed_admissible_count=1,
        ),
        explicit_selection=_selection(release),
        recorded_at=datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc),
    )
    with pytest.raises(CapabilityHandoffConsumerError) as exc_info:
        consumer.consume(envelope)
    assert exc_info.value.disposition is CapabilityHandoffConsumerFailureDisposition.BLOCKED


def test_non_tool_envelope_is_blocked() -> None:
    consumer, _repo = _consumer()
    with pytest.raises(CapabilityHandoffConsumerError) as exc_info:
        consumer.consume(_envelope(_agent_release()))
    assert exc_info.value.disposition is CapabilityHandoffConsumerFailureDisposition.BLOCKED


def test_repository_unavailable_maps_to_unavailable() -> None:
    class _UnavailableRepo:
        def stage(
            self,
            record: MarketplaceQualifiedToolStage,
        ) -> MarketplaceQualifiedToolStageWriteResult:
            raise MarketplaceQualifiedToolStageUnavailableError("down")

        def get(
            self,
            *,
            tenant_id: str,
            handoff_id: str,
        ) -> MarketplaceQualifiedToolStage | None:
            return None

    consumer = ToolQualificationStagingConsumer(_UnavailableRepo())
    with pytest.raises(CapabilityHandoffConsumerError) as exc_info:
        consumer.consume(_envelope())
    assert exc_info.value.disposition is CapabilityHandoffConsumerFailureDisposition.UNAVAILABLE


def test_repository_integrity_failure_maps_to_failed() -> None:
    class _IntegrityRepo:
        def stage(
            self,
            record: MarketplaceQualifiedToolStage,
        ) -> MarketplaceQualifiedToolStageWriteResult:
            raise MarketplaceQualifiedToolStageIntegrityError("corrupt")

        def get(
            self,
            *,
            tenant_id: str,
            handoff_id: str,
        ) -> MarketplaceQualifiedToolStage | None:
            return None

    consumer = ToolQualificationStagingConsumer(_IntegrityRepo())
    with pytest.raises(CapabilityHandoffConsumerError) as exc_info:
        consumer.consume(_envelope())
    assert exc_info.value.disposition is CapabilityHandoffConsumerFailureDisposition.FAILED


def test_consumer_has_stable_consumer_id() -> None:
    consumer, _repo = _consumer()
    assert consumer.consumer_id == TOOL_QUALIFICATION_STAGING_CONSUMER_ID


def test_consumer_has_no_activation_or_execution_dependencies() -> None:
    path = Path(
        "intergrax/marketplace/handoff/adapters/tool_qualification_staging_consumer.py",
    )
    source = path.read_text(encoding="utf-8")
    forbidden = (
        "ToolRegistry",
        "ToolRuntime",
        "DynamicToolAcquisition",
        "CapabilityQualification",
        "QualifiedCapabilityBinding",
    )
    for token in forbidden:
        assert token not in source
