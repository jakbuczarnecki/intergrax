# © Artur Czarnecki. All rights reserved.

"""SKILL-X Batch A — host lifecycle install vs enablement, idempotency, isolation."""

from __future__ import annotations

import pytest

from intergrax.contracts.capability_catalog import (
    CapabilityDiscoveryIdentity,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.lifecycle_handoff.ack import DomainLifecycleHandoffDisposition
from intergrax.skills.catalog import SkillCatalogProviderRegistry, SkillPackageResolution
from intergrax.skills.dynamic_acquisition import (
    DynamicSkillAcquisitionRequest,
    DynamicSkillAcquisitionService,
    SkillHostBindingMaterializer,
)
from intergrax.skills.errors import (
    DynamicSkillAcquisitionBindingError,
    DynamicSkillAcquisitionResolutionError,
)
from intergrax.skills.host_lifecycle import SkillHostLifecycleService
from intergrax.skills.identity import SkillDiscoveryCandidateIdentity, SkillPackageCandidate
from intergrax.skills.registry.factory import enabled_skill_ids_for_profile
from intergrax.skills.registry.profile import SkillProfile, is_skill_enabled
from intergrax.skills.registry.provenance import SkillRuntimeBindingMetadata
from intergrax.skills.registry.runtime import SkillRegistry
from testing_support.canonical_me15_reference_skill import (
    ME15_DIGEST_V1,
    ME15_DIGEST_V2,
    ME15_PACKAGE_REFERENCE_V1,
    ME15_SKILL_LOGICAL_ID,
    ME15_VERSION_V1,
    ME15_VERSION_V2,
)
from testing_support.me15_skill_binding_materializer import Me15SkillHostBindingMaterializer
from testing_support.me15_skill_catalog_provider import Me15SkillCatalogProvider

pytestmark = pytest.mark.unit


def _identity_key(provider: Me15SkillCatalogProvider) -> CapabilityIdentityKey:
    return CapabilityIdentityKey.from_discovery_identity(
        CapabilityDiscoveryIdentity(
            kind=CapabilityKind.SKILL,
            source=CapabilitySourceIdentity(
                source_id=provider.catalog_source_id,
                source_kind=CapabilitySourceKind.OFFICIAL,
            ),
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.SKILL,
                logical_id=ME15_SKILL_LOGICAL_ID,
            ),
        ),
    )


def _candidate(
    *,
    version: str = ME15_VERSION_V1,
    digest: str = ME15_DIGEST_V1,
    provider: Me15SkillCatalogProvider,
) -> SkillDiscoveryCandidateIdentity:
    return SkillDiscoveryCandidateIdentity(
        catalog_source_id=provider.catalog_source_id,
        package=SkillPackageCandidate(
            logical_skill_id=ME15_SKILL_LOGICAL_ID,
            package_reference=ME15_PACKAGE_REFERENCE_V1,
            package_version=version,
            package_digest=digest,
        ),
    )


def _acquisition_stack(
    host_profile_id: str = "host-batch-a",
) -> tuple[SkillHostLifecycleService, DynamicSkillAcquisitionService, Me15SkillCatalogProvider]:
    lifecycle = SkillHostLifecycleService(host_profile_id=host_profile_id)
    provider = Me15SkillCatalogProvider()
    service = DynamicSkillAcquisitionService(
        catalog_registry=SkillCatalogProviderRegistry({provider.catalog_source_id: provider}),
        binding=lifecycle,
        materializer=Me15SkillHostBindingMaterializer(
            lifecycle.registry,
            catalog_source_id=provider.catalog_source_id,
        ),
    )
    return lifecycle, service, provider


def test_install_without_profile_enablement() -> None:
    lifecycle, service, provider = _acquisition_stack()
    service.acquire(
        DynamicSkillAcquisitionRequest(
            operation_id="op-install-only",
            host_profile_id=lifecycle.host_profile_id,
            capability_identity_key=_identity_key(provider),
            selected_identity=_candidate(provider=provider),
        ),
    )
    assert lifecycle.is_bound(ME15_SKILL_LOGICAL_ID)
    assert not is_skill_enabled(lifecycle.skill_profile, ME15_SKILL_LOGICAL_ID)


def test_explicit_enablement_after_install() -> None:
    lifecycle, service, provider = _acquisition_stack()
    service.acquire(
        DynamicSkillAcquisitionRequest(
            operation_id="op-then-enable",
            host_profile_id=lifecycle.host_profile_id,
            capability_identity_key=_identity_key(provider),
            selected_identity=_candidate(provider=provider),
        ),
    )
    lifecycle.enable_installed_skill(ME15_SKILL_LOGICAL_ID)
    assert is_skill_enabled(lifecycle.skill_profile, ME15_SKILL_LOGICAL_ID)


def test_idempotent_replay_when_installed_but_disabled() -> None:
    lifecycle, service, provider = _acquisition_stack()
    request = DynamicSkillAcquisitionRequest(
        operation_id="op-idem-disabled",
        host_profile_id=lifecycle.host_profile_id,
        capability_identity_key=_identity_key(provider),
        selected_identity=_candidate(provider=provider),
    )
    first = service.acquire(request)
    assert not is_skill_enabled(lifecycle.skill_profile, ME15_SKILL_LOGICAL_ID)
    second = service.acquire(request)
    assert second.operation_id == first.operation_id
    assert not is_skill_enabled(lifecycle.skill_profile, ME15_SKILL_LOGICAL_ID)


def test_duplicate_operation_id_rejected() -> None:
    lifecycle, service, provider = _acquisition_stack()
    service.acquire(
        DynamicSkillAcquisitionRequest(
            operation_id="op-dup",
            host_profile_id=lifecycle.host_profile_id,
            capability_identity_key=_identity_key(provider),
            selected_identity=_candidate(provider=provider),
        ),
    )
    with pytest.raises(DynamicSkillAcquisitionBindingError):
        service.acquire(
            DynamicSkillAcquisitionRequest(
                operation_id="op-other",
                host_profile_id=lifecycle.host_profile_id,
                capability_identity_key=_identity_key(provider),
                selected_identity=_candidate(provider=provider),
            ),
        )


def test_host_profile_mismatch_rejected() -> None:
    lifecycle, service, provider = _acquisition_stack(host_profile_id="host-a")
    with pytest.raises(DynamicSkillAcquisitionResolutionError):
        service.acquire(
            DynamicSkillAcquisitionRequest(
                operation_id="op-host",
                host_profile_id="host-b",
                capability_identity_key=_identity_key(provider),
                selected_identity=_candidate(provider=provider),
            ),
        )


def test_cross_host_profile_isolation() -> None:
    life_a, service_a, provider = _acquisition_stack(host_profile_id="tenant-a-host")
    life_b = SkillHostLifecycleService(host_profile_id="tenant-b-host")
    service_b = DynamicSkillAcquisitionService(
        catalog_registry=SkillCatalogProviderRegistry({provider.catalog_source_id: provider}),
        binding=life_b,
        materializer=Me15SkillHostBindingMaterializer(
            life_b.registry,
            catalog_source_id=provider.catalog_source_id,
        ),
    )
    service_a.acquire(
        DynamicSkillAcquisitionRequest(
            operation_id="op-tenant-a",
            host_profile_id=life_a.host_profile_id,
            capability_identity_key=_identity_key(provider),
            selected_identity=_candidate(provider=provider),
        ),
    )
    assert life_a.is_bound(ME15_SKILL_LOGICAL_ID)
    assert not life_b.is_bound(ME15_SKILL_LOGICAL_ID)


def test_register_all_catalog_registers_without_auto_enable() -> None:
    from intergrax.skills.registry.bootstrap import register_default_skills, reset_default_skills_for_tests
    from intergrax.skills.registry.catalog import clear_skill_catalog
    from intergrax.skills.registry.factory import build_registry_from_profile

    reset_default_skills_for_tests()
    clear_skill_catalog()
    register_default_skills()
    profile = SkillProfile(register_all_catalog_bundles=True)
    registry = build_registry_from_profile(profile)
    assert registry.has("legal.contract_review")
    assert not is_skill_enabled(profile, "legal.contract_review")
    assert "legal.contract_review" not in enabled_skill_ids_for_profile(profile)


class _FailingMaterializer(SkillHostBindingMaterializer):
    def materialize(
        self,
        resolution: SkillPackageResolution,
    ) -> tuple[str, SkillRuntimeBindingMetadata]:
        raise RuntimeError("materialize failed")


def test_materializer_failure_does_not_leave_bound_skill() -> None:
    lifecycle = SkillHostLifecycleService(host_profile_id="host-fail")
    provider = Me15SkillCatalogProvider()
    entry = provider.list_entries()[0]
    resolution = provider.resolve_package(entry, version_selector=ME15_VERSION_V1)
    ack = lifecycle.bind(
        operation_id="op-fail",
        host_profile_id=lifecycle.host_profile_id,
        resolved=resolution,
        materializer=_FailingMaterializer(),
    )
    assert ack.disposition is DomainLifecycleHandoffDisposition.REJECTED
    assert not lifecycle.is_bound(ME15_SKILL_LOGICAL_ID)


def test_incompatible_digest_rejected_on_second_bind() -> None:
    lifecycle, service, provider = _acquisition_stack()
    service.acquire(
        DynamicSkillAcquisitionRequest(
            operation_id="op-v1",
            host_profile_id=lifecycle.host_profile_id,
            capability_identity_key=_identity_key(provider),
            selected_identity=_candidate(provider=provider),
        ),
    )
    with pytest.raises(DynamicSkillAcquisitionResolutionError):
        service.acquire(
            DynamicSkillAcquisitionRequest(
                operation_id="op-v2-digest",
                host_profile_id=lifecycle.host_profile_id,
                capability_identity_key=_identity_key(provider),
                selected_identity=_candidate(
                    provider=provider,
                    version=ME15_VERSION_V1,
                    digest=ME15_DIGEST_V2,
                ),
            ),
        )
