# © Artur Czarnecki. All rights reserved.

"""CAPABILITY-CATALOG-1 Stage 13 usage event contract tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from intergrax.contracts.capability_catalog import (
    CapabilityIdentityKey,
    CapabilityKind,
    CapabilityProvenance,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.capability_metering import (
    SCHEMA_CAPABILITY_USAGE_EVENT_V1,
    CapabilityUsageEvent,
    CapabilityUsageKind,
    CapabilityUsageOutcome,
    build_capability_usage_event,
)
from intergrax.contracts.execution_identity import mint_event_id

pytestmark = pytest.mark.unit


def _source(
    *,
    source_id: str = "marketplace.x",
    source_kind: CapabilitySourceKind = CapabilitySourceKind.OFFICIAL,
) -> CapabilitySourceIdentity:
    return CapabilitySourceIdentity(source_id=source_id, source_kind=source_kind)


def _identity(
    *,
    kind: CapabilityKind = CapabilityKind.TOOL,
    source_id: str = "marketplace.x",
    source_kind: CapabilitySourceKind = CapabilitySourceKind.OFFICIAL,
    logical_id: str = "tool.search",
) -> CapabilityIdentityKey:
    return CapabilityIdentityKey(
        kind=kind,
        source_id=source_id,
        source_kind=source_kind,
        logical_id=logical_id,
    )


def _provenance(
    *,
    source_id: str = "marketplace.x",
    source_kind: CapabilitySourceKind = CapabilitySourceKind.OFFICIAL,
    version_label: str | None = "2.1",
    content_digest: str | None = "sha256:abc",
    publisher: str | None = "vendor-x",
) -> CapabilityProvenance:
    return CapabilityProvenance(
        source=_source(source_id=source_id, source_kind=source_kind),
        version_label=version_label,
        content_digest=content_digest,
        publisher=publisher,
    )


def test_capability_usage_event_contract_is_frozen_and_forbids_extra_fields() -> None:
    event = build_capability_usage_event(
        tenant_id="tenant-a",
        identity=_identity(),
        provenance=_provenance(),
        usage_kind=CapabilityUsageKind.EXECUTION,
        outcome=CapabilityUsageOutcome.SUCCEEDED,
    )
    assert event.schema_version == SCHEMA_CAPABILITY_USAGE_EVENT_V1
    with pytest.raises(ValidationError):
        event.tenant_id = "tenant-b"
    with pytest.raises(ValidationError):
        CapabilityUsageEvent.model_validate(
            {
                **event.model_dump(mode="json"),
                "unexpected": "field",
            },
        )


def test_capability_usage_event_rejects_skill_direct_usage() -> None:
    with pytest.raises(ValidationError, match="CapabilityKind.SKILL"):
        build_capability_usage_event(
            tenant_id="tenant-a",
            identity=_identity(
                kind=CapabilityKind.SKILL,
                logical_id="skill.browser",
            ),
            provenance=_provenance(),
            usage_kind=CapabilityUsageKind.EXECUTION,
            outcome=CapabilityUsageOutcome.SUCCEEDED,
        )


def test_capability_usage_event_rejects_unknown_source_kind() -> None:
    with pytest.raises(ValidationError, match="UNKNOWN"):
        build_capability_usage_event(
            tenant_id="tenant-a",
            identity=_identity(source_kind=CapabilitySourceKind.UNKNOWN),
            provenance=_provenance(source_kind=CapabilitySourceKind.UNKNOWN),
            usage_kind=CapabilityUsageKind.EXECUTION,
            outcome=CapabilityUsageOutcome.SUCCEEDED,
        )


def test_capability_usage_event_rejects_identity_provenance_source_mismatch() -> None:
    with pytest.raises(ValidationError, match="source_id"):
        build_capability_usage_event(
            tenant_id="tenant-a",
            identity=_identity(source_id="official.catalog"),
            provenance=_provenance(source_id="enterprise.private"),
            usage_kind=CapabilityUsageKind.EXECUTION,
            outcome=CapabilityUsageOutcome.SUCCEEDED,
        )
    with pytest.raises(ValidationError, match="source_kind"):
        build_capability_usage_event(
            tenant_id="tenant-a",
            identity=_identity(source_kind=CapabilitySourceKind.OFFICIAL),
            provenance=_provenance(source_kind=CapabilitySourceKind.ENTERPRISE_PRIVATE),
            usage_kind=CapabilityUsageKind.EXECUTION,
            outcome=CapabilityUsageOutcome.SUCCEEDED,
        )


def test_capability_usage_event_validates_event_id_and_tenant() -> None:
    with pytest.raises(ValidationError, match="EventId"):
        build_capability_usage_event(
            tenant_id="tenant-a",
            identity=_identity(),
            provenance=_provenance(),
            usage_kind=CapabilityUsageKind.EXECUTION,
            outcome=CapabilityUsageOutcome.SUCCEEDED,
            event_id="not-an-event-id",  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError, match="tenant_id"):
        build_capability_usage_event(
            tenant_id="  ",
            identity=_identity(),
            provenance=_provenance(),
            usage_kind=CapabilityUsageKind.EXECUTION,
            outcome=CapabilityUsageOutcome.SUCCEEDED,
        )


def test_capability_usage_event_requires_timezone_aware_recorded_at() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        build_capability_usage_event(
            tenant_id="tenant-a",
            identity=_identity(),
            provenance=_provenance(),
            usage_kind=CapabilityUsageKind.EXECUTION,
            outcome=CapabilityUsageOutcome.SUCCEEDED,
            recorded_at=datetime(2026, 9, 7, 12, 0, 0),
        )


def test_capability_usage_event_round_trip_preserves_attribution() -> None:
    event = build_capability_usage_event(
        tenant_id="tenant-a",
        identity=_identity(),
        provenance=_provenance(),
        usage_kind=CapabilityUsageKind.EXECUTION,
        outcome=CapabilityUsageOutcome.SUCCEEDED,
        event_id=mint_event_id(),
        recorded_at=datetime(2026, 9, 7, 12, 0, 0, tzinfo=timezone.utc),
    )
    restored = CapabilityUsageEvent.model_validate(event.model_dump(mode="json"))
    assert restored == event
    assert restored.identity.source_kind is CapabilitySourceKind.OFFICIAL
    assert restored.provenance.publisher == "vendor-x"
    assert restored.provenance.version_label == "2.1"
    assert restored.provenance.content_digest == "sha256:abc"
