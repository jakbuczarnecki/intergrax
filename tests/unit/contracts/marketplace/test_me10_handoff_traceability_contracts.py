# © Artur Czarnecki. All rights reserved.

"""ME-10 handoff traceability contract strictness."""

from __future__ import annotations

from datetime import datetime, timezone

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
    CapabilityHandoffEnvelope,
    CapabilityMarketplaceExplicitSelection,
    MarketplaceQueryContext,
)
from intergrax.contracts.marketplace.handoff_traceability import (
    CapabilityHandoffConsumerTarget,
)

pytestmark = pytest.mark.unit

_SOURCE = CapabilitySourceIdentity(
    source_id="official.contracts.me10",
    source_kind=CapabilitySourceKind.OFFICIAL,
)


def _release(version_label: str = "1.0.0") -> CapabilityReleaseIdentity:
    return CapabilityReleaseIdentity(
        discovery=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=_SOURCE,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id="tools.contract",
            ),
        ),
        publisher="pub",
        version_label=version_label,
        content_digest="sha256:contract",
    )


def _trace(discovery_correlation_id: str = "discovery-1") -> CapabilityDiscoveryTraceFacts:
    return CapabilityDiscoveryTraceFacts(
        discovery_correlation_id=discovery_correlation_id,
        marketplace_query_context=MarketplaceQueryContext(tenant_id="tenant-1"),
        visible_candidate_count=1,
        governed_admissible_count=1,
    )


def _selection(release: CapabilityReleaseIdentity) -> CapabilityMarketplaceExplicitSelection:
    return CapabilityMarketplaceExplicitSelection(
        selection_id="selection-1",
        discovery_correlation_id="discovery-1",
        selected_release=release,
        selector_id="selector-1",
        listing_id="listing-1",
    )


def _envelope(release: CapabilityReleaseIdentity | None = None) -> CapabilityHandoffEnvelope:
    release = release or _release()
    return CapabilityHandoffEnvelope(
        handoff_id="handoff-1",
        tenant_id="tenant-1",
        selected_release=release,
        discovery_correlation_id="discovery-1",
        selection_id="selection-1",
        consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
        downstream_consumer_id="consumer-1",
        discovery_trace=_trace(),
        explicit_selection=_selection(release),
        recorded_at=datetime(2026, 3, 16, 12, 0, tzinfo=timezone.utc),
    )


def test_handoff_envelope_json_roundtrip() -> None:
    envelope = _envelope()
    restored = CapabilityHandoffEnvelope.model_validate(envelope.model_dump(mode="json"))
    assert restored == envelope


def test_handoff_envelope_rejects_unknown_fields() -> None:
    payload = _envelope().model_dump(mode="json")
    payload["unexpected"] = True
    with pytest.raises(Exception):
        CapabilityHandoffEnvelope.model_validate(payload)


def test_handoff_envelope_rejects_mismatched_selected_release() -> None:
    release_a = _release("1.0.0")
    release_b = _release("2.0.0")
    with pytest.raises(ValueError, match="selected_release must match explicit_selection"):
        CapabilityHandoffEnvelope(
            handoff_id="handoff-1",
            tenant_id="tenant-1",
            selected_release=release_a,
            discovery_correlation_id="discovery-1",
            selection_id="selection-1",
            consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
            downstream_consumer_id="consumer-1",
            discovery_trace=_trace(),
            explicit_selection=_selection(release_b),
            recorded_at=datetime(2026, 3, 16, 12, 0, tzinfo=timezone.utc),
        )


def test_handoff_envelope_rejects_tenant_mismatch_with_discovery_context() -> None:
    release = _release()
    with pytest.raises(ValueError, match="tenant_id must match marketplace_query_context.tenant_id"):
        CapabilityHandoffEnvelope(
            handoff_id="handoff-1",
            tenant_id=None,
            selected_release=release,
            discovery_correlation_id="discovery-1",
            selection_id="selection-1",
            consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
            downstream_consumer_id="consumer-1",
            discovery_trace=_trace(),
            explicit_selection=_selection(release),
            recorded_at=datetime(2026, 3, 16, 12, 0, tzinfo=timezone.utc),
        )


def test_handoff_envelope_requires_timezone_aware_recorded_at() -> None:
    release = _release()
    with pytest.raises(ValueError, match="recorded_at must be timezone-aware"):
        CapabilityHandoffEnvelope(
            handoff_id="handoff-1",
            tenant_id="tenant-1",
            selected_release=release,
            discovery_correlation_id="discovery-1",
            selection_id="selection-1",
            consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
            downstream_consumer_id="consumer-1",
            discovery_trace=_trace(),
            explicit_selection=_selection(release),
            recorded_at=datetime(2026, 3, 16, 12, 0),
        )
