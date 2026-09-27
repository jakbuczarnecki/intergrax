# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.contracts.capability_catalog import (
    CapabilityDiscoveryIdentity,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityReleaseIdentity,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.marketplace.handoff_traceability import CapabilityHandoffConsumerTarget
from intergrax.contracts.tools.marketplace_qualified_capability import (
    MarketplaceQualifiedToolStage,
)
from intergrax.tools.catalog import ToolCatalogProviderRegistry
from intergrax.tools.dynamic_acquisition import (
    DynamicToolAcquisitionRequest,
    DynamicToolAcquisitionService,
)
from intergrax.tools.host_lifecycle import ToolHostLifecycleService
from intergrax.tools.identity import ToolDiscoveryCandidateIdentity, ToolPackageCandidate
from intergrax.tools.qualified_marketplace_tool_activation_resolver import (
    QualifiedMarketplaceToolActivationOutcome,
    QualifiedMarketplaceToolActivationResolver,
)
from testing_support.canonical_me14_echo_tool import (
    ME14_DIGEST_V1,
    ME14_PACKAGE_REFERENCE_V1,
    ME14_TOOL_LOGICAL_ID,
    ME14_VERSION_V1,
)
from testing_support.me14_tool_activation_materializer import Me14ToolHostActivationMaterializer
from testing_support.me14_tool_catalog_provider import Me14ToolCatalogProvider

pytestmark = pytest.mark.unit

_HOST = "host-profile-me14"
_SOURCE = CapabilitySourceIdentity(
    source_id="official.intergrax.me14",
    source_kind=CapabilitySourceKind.OFFICIAL,
)


def _stage(
    *,
    version_label: str = ME14_VERSION_V1,
    content_digest: str = ME14_DIGEST_V1,
    package_reference: str = ME14_PACKAGE_REFERENCE_V1,
) -> MarketplaceQualifiedToolStage:
    return MarketplaceQualifiedToolStage(
        handoff_id="handoff-me14",
        tenant_id="tenant-a",
        selected_release=CapabilityReleaseIdentity(
            discovery=CapabilityDiscoveryIdentity(
                kind=CapabilityKind.TOOL,
                source=_SOURCE,
                logical=CapabilityLogicalIdentity(
                    kind=CapabilityKind.TOOL,
                    logical_id=ME14_TOOL_LOGICAL_ID,
                ),
            ),
            publisher="publisher:me14",
            version_label=version_label,
            content_digest=content_digest,
            package_reference=package_reference,
        ),
        discovery_correlation_id="disc-1",
        selection_id="sel-1",
        consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
        downstream_consumer_id="tool.qualification_staging.v1",
        recorded_at=datetime(2026, 3, 26, 12, 0, tzinfo=UTC),
    )


def _resolver() -> tuple[QualifiedMarketplaceToolActivationResolver, DynamicToolAcquisitionService]:
    lifecycle = ToolHostLifecycleService(host_profile_id=_HOST)
    provider = Me14ToolCatalogProvider()
    materializer = Me14ToolHostActivationMaterializer(
        lifecycle.registry,
        catalog_source_id=provider.catalog_source_id,
    )
    acquisition = DynamicToolAcquisitionService(
        catalog_registry=ToolCatalogProviderRegistry(
            {provider.catalog_source_id: provider},
        ),
        activation=lifecycle,
        materializer=materializer,
    )
    resolver = QualifiedMarketplaceToolActivationResolver(
        activation_read=lifecycle,
        acquisition=acquisition,
        host_profile_id=_HOST,
    )
    return resolver, acquisition


def test_inactive_exact_activates_once() -> None:
    resolver, _ = _resolver()
    result = resolver.ensure_exact_active(
        stage=_stage(),
        execution_request_id="exec-a",
    )
    assert result.outcome is QualifiedMarketplaceToolActivationOutcome.ACTIVATED_EXACT
    assert result.registry_tool_id == ME14_TOOL_LOGICAL_ID


def test_already_active_exact_reuse_no_second_activation() -> None:
    resolver, _ = _resolver()
    first = resolver.ensure_exact_active(stage=_stage(), execution_request_id="exec-a")
    assert first.outcome is QualifiedMarketplaceToolActivationOutcome.ACTIVATED_EXACT
    second = resolver.ensure_exact_active(stage=_stage(), execution_request_id="exec-b")
    assert second.outcome is QualifiedMarketplaceToolActivationOutcome.ALREADY_ACTIVE_EXACT

def test_already_active_different_version_fails() -> None:
    resolver, _ = _resolver()
    resolver.ensure_exact_active(stage=_stage(), execution_request_id="exec-a")
    conflict = resolver.ensure_exact_active(
        stage=_stage(version_label=ME14_VERSION_V1 + "-other"),
        execution_request_id="exec-b",
    )
    assert conflict.outcome is QualifiedMarketplaceToolActivationOutcome.RELEASE_CONFLICT


def test_missing_staged_digest_fails() -> None:
    resolver, _ = _resolver()
    stage = _stage()
    bad_release = stage.selected_release.model_copy(update={"content_digest": None})
    bad_stage = stage.model_copy(update={"selected_release": bad_release})
    result = resolver.ensure_exact_active(stage=bad_stage, execution_request_id="exec-a")
    assert result.outcome is QualifiedMarketplaceToolActivationOutcome.INTEGRITY_FAILURE
