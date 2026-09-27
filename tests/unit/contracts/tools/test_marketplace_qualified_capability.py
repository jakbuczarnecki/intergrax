# © Artur Czarnecki. All rights reserved.

"""S24-GAP-02-P1 marketplace qualified tool stage contract tests."""

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
from intergrax.contracts.marketplace.handoff_traceability import (
    CapabilityHandoffConsumerTarget,
)
from intergrax.contracts.tools.marketplace_qualified_capability import (
    MarketplaceQualifiedToolStage,
)

pytestmark = pytest.mark.unit

_SOURCE = CapabilitySourceIdentity(
    source_id="official.contracts.gap02",
    source_kind=CapabilitySourceKind.OFFICIAL,
)


def _tool_release(version_label: str = "1.0.0") -> CapabilityReleaseIdentity:
    return CapabilityReleaseIdentity(
        discovery=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=_SOURCE,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id="tools.gap02",
            ),
        ),
        publisher="publisher-gap02",
        version_label=version_label,
        content_digest="sha256:gap02",
        package_reference="pkg://gap02/tool",
    )


def _agent_release() -> CapabilityReleaseIdentity:
    return CapabilityReleaseIdentity(
        discovery=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.AGENT,
            source=_SOURCE,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.AGENT,
                logical_id="agents.gap02",
            ),
        ),
        publisher="publisher-gap02",
        version_label="1.0.0",
    )


def _stage(
    release: CapabilityReleaseIdentity | None = None,
    *,
    consumer_target: CapabilityHandoffConsumerTarget = (
        CapabilityHandoffConsumerTarget.TOOL_DOMAIN
    ),
) -> MarketplaceQualifiedToolStage:
    release = release or _tool_release()
    return MarketplaceQualifiedToolStage(
        handoff_id="handoff-gap02",
        tenant_id="tenant-gap02",
        selected_release=release,
        discovery_correlation_id="discovery-gap02",
        selection_id="selection-gap02",
        consumer_target=consumer_target,
        downstream_consumer_id="tool.qualification_staging.v1",
        recorded_at=datetime(2026, 3, 26, 10, 0, tzinfo=timezone.utc),
    )


def test_valid_tool_stage_accepted() -> None:
    stage = _stage()
    assert stage.schema_version == "marketplace_qualified_tool_stage.v1"


def test_stage_model_is_frozen() -> None:
    stage = _stage()
    with pytest.raises(Exception):
        stage.handoff_id = "other"  # type: ignore[misc]


def test_stage_rejects_extra_fields() -> None:
    payload = _stage().model_dump(mode="json")
    payload["metadata"] = {"unexpected": True}
    with pytest.raises(Exception):
        MarketplaceQualifiedToolStage.model_validate(payload)


def test_stage_rejects_empty_handoff_id() -> None:
    payload = _stage().model_dump()
    payload["handoff_id"] = ""
    with pytest.raises(ValueError):
        MarketplaceQualifiedToolStage.model_validate(payload)


def test_stage_requires_tenant_id() -> None:
    with pytest.raises(ValueError):
        MarketplaceQualifiedToolStage(
            handoff_id="handoff-gap02",
            tenant_id="",
            selected_release=_tool_release(),
            discovery_correlation_id="discovery-gap02",
            selection_id="selection-gap02",
            consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
            downstream_consumer_id="tool.qualification_staging.v1",
            recorded_at=datetime(2026, 3, 26, 10, 0, tzinfo=timezone.utc),
        )


def test_stage_rejects_wrong_consumer_target() -> None:
    with pytest.raises(ValueError, match="TOOL_DOMAIN"):
        _stage(consumer_target=CapabilityHandoffConsumerTarget.AGENT_DOMAIN)


def test_stage_rejects_non_tool_release() -> None:
    with pytest.raises(ValueError, match="TOOL"):
        _stage(release=_agent_release())


def test_stage_rejects_naive_recorded_at() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        MarketplaceQualifiedToolStage(
            handoff_id="handoff-gap02",
            tenant_id="tenant-gap02",
            selected_release=_tool_release(),
            discovery_correlation_id="discovery-gap02",
            selection_id="selection-gap02",
            consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
            downstream_consumer_id="tool.qualification_staging.v1",
            recorded_at=datetime(2026, 3, 26, 10, 0),
        )


def test_stage_preserves_exact_release_identity() -> None:
    release = _tool_release("9.9.9")
    stage = _stage(release)
    assert stage.selected_release == release
    assert stage.selected_release.publisher == "publisher-gap02"
    assert stage.selected_release.content_digest == "sha256:gap02"
    assert stage.selected_release.package_reference == "pkg://gap02/tool"


def test_stage_json_roundtrip_preserves_release() -> None:
    stage = _stage(_tool_release("2.1.0"))
    restored = MarketplaceQualifiedToolStage.model_validate(
        stage.model_dump(mode="json"),
    )
    assert restored == stage
    assert restored.selected_release.version_label == "2.1.0"
