# © Artur Czarnecki. All rights reserved.

"""UCA-2 — Tool domain realization handoff and availability projection."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.capability_acquisition.adapters.tool import ToolCapabilityRealizationProvider
from intergrax.capability_acquisition.availability_projection import (
    project_availability_disposition,
)
from intergrax.capability_acquisition.service import CapabilityRealizationService
from intergrax.contracts.capability_acquisition.outcome import (
    CapabilityRealizationOutcome,
)
from intergrax.contracts.capability_acquisition.request import (
    CapabilityRealizationRequest,
    derive_capability_realization_request_id,
)
from intergrax.contracts.capability_catalog.availability import AvailabilityDisposition
from intergrax.contracts.capability_catalog.capability_realization_need import (
    CapabilityRealizationNeed,
)
from intergrax.contracts.capability_catalog.discovery_completion import (
    build_discovery_completion,
)
from intergrax.contracts.capability_catalog.federation import (
    CapabilityCatalogFederationCompleteness,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.identity import CapabilitySourceKind
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.tools.known_capability_realization import (
    KnownToolCapabilityRealizationRequest,
)
from intergrax.tools.catalog import ToolCatalogEntry, ToolPackageCandidate, ToolPackageResolution
from intergrax.tools.host_lifecycle import ToolHostLifecycleService
from intergrax.tools.known_capability_realization import ToolKnownCapabilityRealizationService
from testing_support.canonical_me14_echo_tool import (
    ME14_DIGEST_V1,
    ME14_PACKAGE_REFERENCE_V1,
    ME14_TOOL_LOGICAL_ID,
    ME14_VERSION_V1,
)
from testing_support.me14_tool_activation_materializer import Me14ToolHostActivationMaterializer
from testing_support.me14_tool_catalog_provider import Me14ToolCatalogProvider

pytestmark = pytest.mark.unit

_CREATED = datetime(2026, 9, 20, 10, 0, tzinfo=UTC)


def _need() -> CapabilityRealizationNeed:
    provider = Me14ToolCatalogProvider()
    key = CapabilityIdentityKey(
        kind=CapabilityKind.TOOL,
        source_id=provider.catalog_source_id,
        source_kind=CapabilitySourceKind.OFFICIAL,
        logical_id=ME14_TOOL_LOGICAL_ID,
    )
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        suitable_catalog_allowed_keys=(key,),
        created_at=_CREATED,
    )
    return CapabilityRealizationNeed.from_discovery_completion(
        completion,
        capability_identity=key,
    )


def _request(need: CapabilityRealizationNeed) -> CapabilityRealizationRequest:
    return CapabilityRealizationRequest(
        request_id=derive_capability_realization_request_id(
            realization_need_id=need.realization_need_id,
            request_nonce="nonce-1",
        ),
        request_nonce="nonce-1",
        realization_need=need,
        host_profile_id="host-profile-1",
        requested_at=_CREATED,
    )


class _StaticResolver:
    def __init__(self, resolution: ToolPackageResolution) -> None:
        self._resolution = resolution
        self.calls = 0

    def resolve_for_identity(
        self,
        capability_identity: CapabilityIdentityKey,
    ) -> ToolPackageResolution:
        self.calls += 1
        return self._resolution


def _resolution() -> ToolPackageResolution:
    provider = Me14ToolCatalogProvider()
    entry = ToolCatalogEntry(
        catalog_entry_id="entry-me14",
        catalog_source_id=provider.catalog_source_id,
        logical_tool_id=ME14_TOOL_LOGICAL_ID,
        package_reference=ME14_PACKAGE_REFERENCE_V1,
        display_name="ME14 Echo Tool",
    )
    candidate = ToolPackageCandidate(
        logical_tool_id=ME14_TOOL_LOGICAL_ID,
        package_reference=ME14_PACKAGE_REFERENCE_V1,
        package_version=ME14_VERSION_V1,
        package_digest=ME14_DIGEST_V1,
    )
    return ToolPackageResolution(entry=entry, package_candidate=candidate)


def test_tool_handoff_success_projects_host_available() -> None:
    need = _need()
    provider = Me14ToolCatalogProvider()
    lifecycle = ToolHostLifecycleService(host_profile_id="host-profile-1")
    materializer = Me14ToolHostActivationMaterializer(
        lifecycle.registry,
        catalog_source_id=provider.catalog_source_id,
    )
    resolver = _StaticResolver(_resolution())
    domain = ToolKnownCapabilityRealizationService(
        activation=lifecycle,
        materializer=materializer,
        resolver=resolver,
    )
    provider = ToolCapabilityRealizationProvider(domain)
    service = CapabilityRealizationService((provider,))
    result = service.realize(_request(need))
    assert result.outcome is CapabilityRealizationOutcome.SUCCEEDED
    assert result.evidence is not None
    disposition = project_availability_disposition(
        identity=need.capability_identity,
        evidence=result.evidence,
    )
    assert disposition is AvailabilityDisposition.HOST_AVAILABLE


def test_tool_domain_idempotent_operation_id() -> None:
    need = _need()
    provider = Me14ToolCatalogProvider()
    lifecycle = ToolHostLifecycleService(host_profile_id="host-profile-1")
    materializer = Me14ToolHostActivationMaterializer(
        lifecycle.registry,
        catalog_source_id=provider.catalog_source_id,
    )
    resolver = _StaticResolver(_resolution())
    domain = ToolKnownCapabilityRealizationService(
        activation=lifecycle,
        materializer=materializer,
        resolver=resolver,
    )
    domain_request = KnownToolCapabilityRealizationRequest(
        operation_id="op-1",
        host_profile_id="host-profile-1",
        capability_identity=need.capability_identity,
        requested_at=_CREATED,
    )
    first = domain.realize(domain_request)
    second = domain.realize(domain_request)
    assert first is second
    assert resolver.calls == 1
