# © Artur Czarnecki. All rights reserved.

"""ME-10-R2-Q1 — provider lifecycle transition enforcement."""

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
    CapabilityHandoffIdentityConflictError,
    CapabilityMarketplaceExplicitSelection,
    MarketplaceQueryContext,
)
from intergrax.contracts.marketplace.handoff_traceability import (
    CapabilityHandoffConsumerTarget,
    CapabilityHandoffDeliveryAdmissionVerdict,
    CapabilityHandoffDeliveryLifecycleTransitionError,
)
from intergrax.marketplace.handoff_traceability import InMemoryCapabilityHandoffDeliveryAdmission

pytestmark = pytest.mark.unit

_SOURCE = CapabilitySourceIdentity(
    source_id="official.me10r2q1",
    source_kind=CapabilitySourceKind.OFFICIAL,
)


def _release(*, digest: str = "sha256:q1") -> CapabilityReleaseIdentity:
    return CapabilityReleaseIdentity(
        discovery=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=_SOURCE,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id="tools.q1",
            ),
        ),
        publisher="pub",
        version_label="1.0.0",
        content_digest=digest,
    )


def _envelope(
    *,
    handoff_id: str = "handoff-q1",
    tenant_id: str | None = "tenant-a",
    digest: str = "sha256:q1",
) -> CapabilityHandoffEnvelope:
    release = _release(digest=digest)
    return CapabilityHandoffEnvelope(
        handoff_id=handoff_id,
        tenant_id=tenant_id,
        selected_release=release,
        discovery_correlation_id="discovery-1",
        selection_id="selection-1",
        consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
        downstream_consumer_id="consumer-B",
        discovery_trace=CapabilityDiscoveryTraceFacts(
            discovery_correlation_id="discovery-1",
            marketplace_query_context=MarketplaceQueryContext(tenant_id=tenant_id),
            visible_candidate_count=1,
            governed_admissible_count=1,
        ),
        explicit_selection=CapabilityMarketplaceExplicitSelection(
            selection_id="selection-1",
            discovery_correlation_id="discovery-1",
            selected_release=release,
            selector_id="selector-1",
        ),
        recorded_at=datetime(2026, 3, 16, 12, 0, tzinfo=timezone.utc),
    )


def _admission_with_in_progress(envelope: CapabilityHandoffEnvelope) -> InMemoryCapabilityHandoffDeliveryAdmission:
    admission = InMemoryCapabilityHandoffDeliveryAdmission()
    verdict = admission.reserve(envelope)
    assert verdict.verdict is CapabilityHandoffDeliveryAdmissionVerdict.RESERVED_NEW
    return admission


def test_failed_retryable_mark_delivered_rejected_state_unchanged() -> None:
    envelope = _envelope()
    admission = _admission_with_in_progress(envelope)
    admission.mark_delivery_failed(envelope.handoff_id)
    with pytest.raises(CapabilityHandoffDeliveryLifecycleTransitionError) as exc_info:
        admission.mark_delivered(envelope.handoff_id)
    message = str(exc_info.value).lower()
    assert envelope.handoff_id in str(exc_info.value)
    assert "in_progress" in message or "failed_retryable" in message
    retry = admission.reserve(envelope)
    assert retry.verdict is CapabilityHandoffDeliveryAdmissionVerdict.RESERVED_NEW


def test_failed_retryable_mark_delivery_failed_again_rejected() -> None:
    envelope = _envelope()
    admission = _admission_with_in_progress(envelope)
    admission.mark_delivery_failed(envelope.handoff_id)
    with pytest.raises(CapabilityHandoffDeliveryLifecycleTransitionError):
        admission.mark_delivery_failed(envelope.handoff_id)
    retry = admission.reserve(envelope)
    assert retry.verdict is CapabilityHandoffDeliveryAdmissionVerdict.RESERVED_NEW


def test_delivered_mark_delivered_rejected() -> None:
    envelope = _envelope()
    admission = _admission_with_in_progress(envelope)
    admission.mark_delivered(envelope.handoff_id)
    with pytest.raises(CapabilityHandoffDeliveryLifecycleTransitionError):
        admission.mark_delivered(envelope.handoff_id)
    assert envelope.handoff_id in admission.delivered_handoff_ids()


def test_delivered_mark_delivery_failed_rejected() -> None:
    envelope = _envelope()
    admission = _admission_with_in_progress(envelope)
    admission.mark_delivered(envelope.handoff_id)
    with pytest.raises(CapabilityHandoffDeliveryLifecycleTransitionError):
        admission.mark_delivery_failed(envelope.handoff_id)
    assert envelope.handoff_id in admission.delivered_handoff_ids()
    dup = admission.reserve(envelope)
    assert dup.verdict is CapabilityHandoffDeliveryAdmissionVerdict.ALREADY_DELIVERED_IDENTICAL


def test_absent_mark_delivered_rejected() -> None:
    admission = InMemoryCapabilityHandoffDeliveryAdmission()
    with pytest.raises(CapabilityHandoffDeliveryLifecycleTransitionError) as exc_info:
        admission.mark_delivered("unknown-handoff")
    assert "unknown-handoff" in str(exc_info.value)


def test_absent_mark_delivery_failed_rejected() -> None:
    admission = InMemoryCapabilityHandoffDeliveryAdmission()
    with pytest.raises(CapabilityHandoffDeliveryLifecycleTransitionError) as exc_info:
        admission.mark_delivery_failed("unknown-handoff")
    assert "unknown-handoff" in str(exc_info.value)


def test_legal_retry_lifecycle_path() -> None:
    envelope = _envelope()
    admission = InMemoryCapabilityHandoffDeliveryAdmission()
    first = admission.reserve(envelope)
    assert first.verdict is CapabilityHandoffDeliveryAdmissionVerdict.RESERVED_NEW
    admission.mark_delivery_failed(envelope.handoff_id)
    second = admission.reserve(envelope)
    assert second.verdict is CapabilityHandoffDeliveryAdmissionVerdict.RESERVED_NEW
    admission.mark_delivered(envelope.handoff_id)
    assert envelope.handoff_id in admission.delivered_handoff_ids()


def test_failed_retryable_conflict_binding_preserved() -> None:
    envelope = _envelope()
    admission = _admission_with_in_progress(envelope)
    admission.mark_delivery_failed(envelope.handoff_id)
    different = _envelope(digest="sha256:other")
    with pytest.raises(CapabilityHandoffIdentityConflictError):
        admission.reserve(different)


def test_delivered_identical_reserve_is_duplicate() -> None:
    envelope = _envelope()
    admission = _admission_with_in_progress(envelope)
    admission.mark_delivered(envelope.handoff_id)
    dup = admission.reserve(envelope)
    assert dup.verdict is CapabilityHandoffDeliveryAdmissionVerdict.ALREADY_DELIVERED_IDENTICAL


def test_in_progress_identical_reserve_skips_new_attempt() -> None:
    envelope = _envelope()
    admission = _admission_with_in_progress(envelope)
    again = admission.reserve(envelope)
    assert again.verdict is CapabilityHandoffDeliveryAdmissionVerdict.IN_PROGRESS_IDENTICAL
