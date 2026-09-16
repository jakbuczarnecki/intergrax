# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.tools.catalog import ToolCatalogProviderRegistry
from intergrax.tools.dynamic_acquisition import (
    DynamicToolAcquisitionService,
    resolve_discovery_candidate_exact,
)
from intergrax.tools.errors import DynamicToolAcquisitionResolutionError
from intergrax.tools.host_lifecycle import ToolHostLifecycleService
from intergrax.tools.identity import ToolDiscoveryCandidateIdentity, ToolPackageCandidate
from testing_support.canonical_me14_echo_tool import (
    ME14_DIGEST_V1,
    ME14_DIGEST_V2,
    ME14_PACKAGE_REFERENCE_V1,
    ME14_TOOL_LOGICAL_ID,
    ME14_VERSION_V1,
)
from testing_support.me14_tool_activation_materializer import Me14ToolHostActivationMaterializer
from testing_support.me14_tool_catalog_provider import Me14ToolCatalogProvider

pytestmark = pytest.mark.unit


def test_resolve_discovery_candidate_exact_selects_v1_not_v2() -> None:
    provider = Me14ToolCatalogProvider()
    registry = ToolCatalogProviderRegistry({provider.catalog_source_id: provider})
    identity = ToolDiscoveryCandidateIdentity(
        catalog_source_id=provider.catalog_source_id,
        package=ToolPackageCandidate(
            logical_tool_id=ME14_TOOL_LOGICAL_ID,
            package_reference=ME14_PACKAGE_REFERENCE_V1,
            package_version=ME14_VERSION_V1,
            package_digest=ME14_DIGEST_V1,
        ),
    )
    resolution = resolve_discovery_candidate_exact(
        identity=identity,
        catalog_entry_id=None,
        registry=registry,
    )
    assert resolution.package_candidate.package_version == ME14_VERSION_V1
    assert resolution.package_candidate.package_digest == ME14_DIGEST_V1


def test_version_mismatch_fails_closed() -> None:
    provider = Me14ToolCatalogProvider()
    registry = ToolCatalogProviderRegistry({provider.catalog_source_id: provider})
    identity = ToolDiscoveryCandidateIdentity(
        catalog_source_id=provider.catalog_source_id,
        package=ToolPackageCandidate(
            logical_tool_id=ME14_TOOL_LOGICAL_ID,
            package_reference=ME14_PACKAGE_REFERENCE_V1,
            package_version=ME14_VERSION_V1,
            package_digest=ME14_DIGEST_V2,
        ),
    )
    with pytest.raises(DynamicToolAcquisitionResolutionError):
        resolve_discovery_candidate_exact(
            identity=identity,
            catalog_entry_id=None,
            registry=registry,
        )


def test_acquisition_idempotent_operation_replay() -> None:
    lifecycle = ToolHostLifecycleService(host_profile_id="host-profile-me14")
    provider = Me14ToolCatalogProvider()
    materializer = Me14ToolHostActivationMaterializer(
        lifecycle.registry,
        catalog_source_id=provider.catalog_source_id,
    )
    service = DynamicToolAcquisitionService(
        catalog_registry=ToolCatalogProviderRegistry(
            {provider.catalog_source_id: provider},
        ),
        activation=lifecycle,
        materializer=materializer,
    )
    from intergrax.contracts.capability_catalog import (
        CapabilityDiscoveryIdentity,
        CapabilityKind,
        CapabilityLogicalIdentity,
        CapabilitySourceIdentity,
        CapabilitySourceKind,
    )
    from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
    from intergrax.tools.dynamic_acquisition import DynamicToolAcquisitionRequest

    identity = ToolDiscoveryCandidateIdentity(
        catalog_source_id=provider.catalog_source_id,
        package=ToolPackageCandidate(
            logical_tool_id=ME14_TOOL_LOGICAL_ID,
            package_reference=ME14_PACKAGE_REFERENCE_V1,
            package_version=ME14_VERSION_V1,
            package_digest=ME14_DIGEST_V1,
        ),
    )
    identity_key = CapabilityIdentityKey.from_discovery_identity(
        CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=CapabilitySourceIdentity(
                source_id=provider.catalog_source_id,
                source_kind=CapabilitySourceKind.OFFICIAL,
            ),
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id=ME14_TOOL_LOGICAL_ID,
            ),
        ),
    )
    request = DynamicToolAcquisitionRequest(
        operation_id="op-idempotent",
        host_profile_id="host-profile-me14",
        capability_identity_key=identity_key,
        selected_identity=identity,
    )
    first = service.acquire(request)
    second = service.acquire(request)
    assert first.resolved_package_identity.package_version == ME14_VERSION_V1
    assert second.operation_id == first.operation_id
