# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P1 adoption and effective identity contract tests."""

from __future__ import annotations

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.existing_capability_configuration import (
    ConfiguredCapabilityBinding,
)
from intergrax.integrations.contracts.execution_integration_configuration import (
    EffectiveIntegrationIdentity,
    ExecutionIntegrationConfigurationAdoption,
    ExecutionIntegrationConfigurationAdoptionError,
    ExecutionIntegrationConfigurationAdoptionFailureReason,
    IntegrationMaterializationKind,
    validate_configured_adoption_match,
    validate_execution_integration_configuration_adoption,
)

pytestmark = pytest.mark.unit


def _binding(
    *,
    tenant_id: str = "tenant-a",
    category: IntegrationCategory = IntegrationCategory.RELATIONAL_STORE,
    provider_id: str = "sqlite",
    resource_scope: str = "scope-a",
    fingerprint: str = "fp-bind-1",
) -> ConfiguredCapabilityBinding:
    return ConfiguredCapabilityBinding(
        tenant_id=tenant_id,
        integration_category=category,
        provider_id=provider_id,
        resource_scope=resource_scope,
        configuration_type="test.config.v1",
        configuration_version="1",
        configuration_fingerprint=fingerprint,
    )


def _adoption(
    binding: ConfiguredCapabilityBinding | None = None,
    *,
    category: IntegrationCategory = IntegrationCategory.RELATIONAL_STORE,
    resource_scope: str = "scope-a",
) -> ExecutionIntegrationConfigurationAdoption:
    b = binding or _binding(resource_scope=resource_scope, category=category)
    return ExecutionIntegrationConfigurationAdoption(
        configured_binding=b,
        integration_category=category,
        resource_scope=resource_scope,
    )


def _effective(
    *,
    category: IntegrationCategory = IntegrationCategory.RELATIONAL_STORE,
    provider_id: str = "sqlite",
    kind: IntegrationMaterializationKind = IntegrationMaterializationKind.CATALOG_FACTORY,
) -> EffectiveIntegrationIdentity:
    return EffectiveIntegrationIdentity(
        integration_category=category,
        provider_id=provider_id,
        materialization_kind=kind,
    )


def test_valid_catalog_factory_identity() -> None:
    identity = _effective()
    assert identity.materialization_kind == IntegrationMaterializationKind.CATALOG_FACTORY


def test_valid_profile_prebuilt_identity_shape() -> None:
    identity = _effective(kind=IntegrationMaterializationKind.PROFILE_PREBUILT)
    assert identity.materialization_kind == IntegrationMaterializationKind.PROFILE_PREBUILT


def test_effective_rejects_empty_provider_id() -> None:
    with pytest.raises(ExecutionIntegrationConfigurationAdoptionError):
        EffectiveIntegrationIdentity(
            integration_category=IntegrationCategory.RELATIONAL_STORE,
            provider_id="",
            materialization_kind=IntegrationMaterializationKind.CATALOG_FACTORY,
        )


def test_match_rejects_tenant_mismatch() -> None:
    adoption = _adoption(_binding(tenant_id="tenant-b"))
    with pytest.raises(ExecutionIntegrationConfigurationAdoptionError) as exc:
        validate_configured_adoption_match(
            adoption=adoption,
            effective=_effective(),
            expected_tenant_id="tenant-a",
        )
    assert (
        exc.value.reason
        == ExecutionIntegrationConfigurationAdoptionFailureReason.CONFIGURED_ADOPTION_TENANT_MISMATCH
    )


def test_match_rejects_category_mismatch() -> None:
    adoption = _adoption(
        _binding(category=IntegrationCategory.VECTOR_STORE),
        category=IntegrationCategory.VECTOR_STORE,
    )
    with pytest.raises(ExecutionIntegrationConfigurationAdoptionError) as exc:
        validate_configured_adoption_match(
            adoption=adoption,
            effective=_effective(category=IntegrationCategory.RELATIONAL_STORE),
            expected_tenant_id="tenant-a",
        )
    assert (
        exc.value.reason
        == ExecutionIntegrationConfigurationAdoptionFailureReason.CONFIGURED_ADOPTION_CATEGORY_MISMATCH
    )


def test_match_rejects_provider_mismatch() -> None:
    adoption = _adoption(_binding(provider_id="postgres"))
    with pytest.raises(ExecutionIntegrationConfigurationAdoptionError) as exc:
        validate_configured_adoption_match(
            adoption=adoption,
            effective=_effective(provider_id="sqlite"),
            expected_tenant_id="tenant-a",
        )
    assert (
        exc.value.reason
        == ExecutionIntegrationConfigurationAdoptionFailureReason.CONFIGURED_ADOPTION_PROVIDER_MISMATCH
    )


def test_match_rejects_resource_scope_mismatch() -> None:
    binding = _binding(resource_scope="scope-a")
    with pytest.raises(ExecutionIntegrationConfigurationAdoptionError) as exc:
        ExecutionIntegrationConfigurationAdoption(
            configured_binding=binding,
            integration_category=binding.integration_category,
            resource_scope="scope-b",
        )
    assert (
        exc.value.reason
        == ExecutionIntegrationConfigurationAdoptionFailureReason.CONFIGURED_ADOPTION_RESOURCE_SCOPE_MISMATCH
    )


def test_valid_exact_match() -> None:
    adoption = _adoption()
    validate_configured_adoption_match(
        adoption=adoption,
        effective=_effective(),
        expected_tenant_id="tenant-a",
    )


def test_configured_fingerprint_not_compared_as_effective_identity() -> None:
    """Different fingerprint still matches when provider/category/scope align."""
    adoption = _adoption(_binding(fingerprint="fp-other"))
    validate_configured_adoption_match(
        adoption=adoption,
        effective=_effective(),
        expected_tenant_id="tenant-a",
    )


def test_external_work_configured_adoption_unsupported() -> None:
    adoption = _adoption(
        _binding(category=IntegrationCategory.EXTERNAL_WORK, provider_id="ext"),
        category=IntegrationCategory.EXTERNAL_WORK,
    )
    with pytest.raises(ExecutionIntegrationConfigurationAdoptionError) as exc:
        validate_configured_adoption_match(
            adoption=adoption,
            effective=_effective(
                category=IntegrationCategory.EXTERNAL_WORK,
                provider_id="ext",
            ),
            expected_tenant_id="tenant-a",
        )
    assert (
        exc.value.reason
        == ExecutionIntegrationConfigurationAdoptionFailureReason.EFFECTIVE_PROVIDER_IDENTITY_UNAVAILABLE
    )


def test_adoption_retains_original_binding_instance() -> None:
    binding = _binding()
    adoption = _adoption(binding)
    assert adoption.configured_binding is binding
