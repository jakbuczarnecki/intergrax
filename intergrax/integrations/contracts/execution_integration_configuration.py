# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Configured adoption and effective integration identity contracts (TRACE-X-P5-R2-P1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.existing_capability_configuration import (
    ConfiguredCapabilityBinding,
)


class IntegrationMaterializationKind(StrEnum):
    CATALOG_FACTORY = "catalog_factory"
    PROFILE_PREBUILT = "profile_prebuilt"


class ExecutionIntegrationConfigurationAdoptionFailureReason(StrEnum):
    CONFIGURED_ADOPTION_TENANT_MISMATCH = "CONFIGURED_ADOPTION_TENANT_MISMATCH"
    CONFIGURED_ADOPTION_CATEGORY_MISMATCH = "CONFIGURED_ADOPTION_CATEGORY_MISMATCH"
    CONFIGURED_ADOPTION_PROVIDER_MISMATCH = "CONFIGURED_ADOPTION_PROVIDER_MISMATCH"
    CONFIGURED_ADOPTION_RESOURCE_SCOPE_MISMATCH = (
        "CONFIGURED_ADOPTION_RESOURCE_SCOPE_MISMATCH"
    )
    EFFECTIVE_PROVIDER_IDENTITY_UNAVAILABLE = "EFFECTIVE_PROVIDER_IDENTITY_UNAVAILABLE"
    CONFIGURED_ADOPTION_REQUIRED_BUT_MISSING = "CONFIGURED_ADOPTION_REQUIRED_BUT_MISSING"


class ExecutionIntegrationConfigurationAdoptionError(Exception):
    """Typed configured-adoption boundary failure."""

    def __init__(
        self,
        reason: ExecutionIntegrationConfigurationAdoptionFailureReason,
        *,
        detail: str = "",
    ) -> None:
        self.reason = reason
        self.detail = detail
        message = reason.value if not detail else f"{reason.value}: {detail}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class EffectiveIntegrationIdentity:
    """Observed effective provider identity — no configured fingerprint or permission."""

    integration_category: IntegrationCategory
    provider_id: str
    materialization_kind: IntegrationMaterializationKind

    def __post_init__(self) -> None:
        validate_effective_integration_identity(self)


@dataclass(frozen=True, slots=True)
class ExecutionIntegrationConfigurationAdoption:
    """Explicit configured binding adoption — not effective provider proof."""

    configured_binding: ConfiguredCapabilityBinding
    integration_category: IntegrationCategory
    resource_scope: str

    def __post_init__(self) -> None:
        validate_execution_integration_configuration_adoption(self)


def validate_effective_integration_identity(
    identity: EffectiveIntegrationIdentity,
) -> None:
    category = identity.integration_category
    if not isinstance(category, IntegrationCategory):
        raise ExecutionIntegrationConfigurationAdoptionError(
            ExecutionIntegrationConfigurationAdoptionFailureReason.CONFIGURED_ADOPTION_CATEGORY_MISMATCH,
            detail="invalid effective integration_category",
        )
    materialization = identity.materialization_kind
    if not isinstance(materialization, IntegrationMaterializationKind):
        raise ExecutionIntegrationConfigurationAdoptionError(
            ExecutionIntegrationConfigurationAdoptionFailureReason.EFFECTIVE_PROVIDER_IDENTITY_UNAVAILABLE,
            detail="invalid materialization_kind",
        )
    provider = identity.provider_id
    if type(provider) is not str or not provider or provider != provider.strip():
        raise ExecutionIntegrationConfigurationAdoptionError(
            ExecutionIntegrationConfigurationAdoptionFailureReason.EFFECTIVE_PROVIDER_IDENTITY_UNAVAILABLE,
            detail="invalid effective provider_id",
        )


def validate_configured_capability_binding_identity(
    binding: ConfiguredCapabilityBinding,
) -> None:
    tenant = binding.tenant_id
    if type(tenant) is not str or not tenant or tenant != tenant.strip():
        raise ExecutionIntegrationConfigurationAdoptionError(
            ExecutionIntegrationConfigurationAdoptionFailureReason.CONFIGURED_ADOPTION_TENANT_MISMATCH,
            detail="invalid binding tenant_id",
        )
    provider = binding.provider_id
    if type(provider) is not str or not provider or provider != provider.strip():
        raise ExecutionIntegrationConfigurationAdoptionError(
            ExecutionIntegrationConfigurationAdoptionFailureReason.CONFIGURED_ADOPTION_PROVIDER_MISMATCH,
            detail="invalid binding provider_id",
        )
    scope = binding.resource_scope
    if type(scope) is not str or not scope or scope != scope.strip():
        raise ExecutionIntegrationConfigurationAdoptionError(
            ExecutionIntegrationConfigurationAdoptionFailureReason.CONFIGURED_ADOPTION_RESOURCE_SCOPE_MISMATCH,
            detail="invalid binding resource_scope",
        )
    config_type = binding.configuration_type
    if type(config_type) is not str or not config_type or config_type != config_type.strip():
        raise ExecutionIntegrationConfigurationAdoptionError(
            ExecutionIntegrationConfigurationAdoptionFailureReason.CONFIGURED_ADOPTION_REQUIRED_BUT_MISSING,
            detail="invalid binding configuration_type",
        )
    config_version = binding.configuration_version
    if (
        type(config_version) is not str
        or not config_version
        or config_version != config_version.strip()
    ):
        raise ExecutionIntegrationConfigurationAdoptionError(
            ExecutionIntegrationConfigurationAdoptionFailureReason.CONFIGURED_ADOPTION_REQUIRED_BUT_MISSING,
            detail="invalid binding configuration_version",
        )
    fingerprint = binding.configuration_fingerprint
    if type(fingerprint) is not str or not fingerprint or fingerprint != fingerprint.strip():
        raise ExecutionIntegrationConfigurationAdoptionError(
            ExecutionIntegrationConfigurationAdoptionFailureReason.CONFIGURED_ADOPTION_REQUIRED_BUT_MISSING,
            detail="invalid binding configuration_fingerprint",
        )
    category = binding.integration_category
    if not isinstance(category, IntegrationCategory):
        raise ExecutionIntegrationConfigurationAdoptionError(
            ExecutionIntegrationConfigurationAdoptionFailureReason.CONFIGURED_ADOPTION_CATEGORY_MISMATCH,
            detail="invalid binding integration_category",
        )


def validate_execution_integration_configuration_adoption(
    adoption: ExecutionIntegrationConfigurationAdoption,
) -> None:
    binding = adoption.configured_binding
    if not isinstance(binding, ConfiguredCapabilityBinding):
        raise ExecutionIntegrationConfigurationAdoptionError(
            ExecutionIntegrationConfigurationAdoptionFailureReason.CONFIGURED_ADOPTION_REQUIRED_BUT_MISSING,
            detail="invalid configured_binding",
        )
    category = adoption.integration_category
    if not isinstance(category, IntegrationCategory):
        raise ExecutionIntegrationConfigurationAdoptionError(
            ExecutionIntegrationConfigurationAdoptionFailureReason.CONFIGURED_ADOPTION_CATEGORY_MISMATCH,
            detail="invalid adoption integration_category",
        )
    validate_configured_capability_binding_identity(binding)
    if binding.integration_category != adoption.integration_category:
        raise ExecutionIntegrationConfigurationAdoptionError(
            ExecutionIntegrationConfigurationAdoptionFailureReason.CONFIGURED_ADOPTION_CATEGORY_MISMATCH,
            detail="binding category mismatch",
        )
    if binding.resource_scope != adoption.resource_scope:
        raise ExecutionIntegrationConfigurationAdoptionError(
            ExecutionIntegrationConfigurationAdoptionFailureReason.CONFIGURED_ADOPTION_RESOURCE_SCOPE_MISMATCH,
            detail="binding resource_scope mismatch",
        )


def validate_configured_adoption_match(
    *,
    adoption: ExecutionIntegrationConfigurationAdoption,
    effective: EffectiveIntegrationIdentity,
    expected_tenant_id: str,
) -> None:
    """Pure configured versus effective match — no lookup or materialization."""
    validate_execution_integration_configuration_adoption(adoption)
    validate_effective_integration_identity(effective)
    if effective.integration_category == IntegrationCategory.EXTERNAL_WORK:
        raise ExecutionIntegrationConfigurationAdoptionError(
            ExecutionIntegrationConfigurationAdoptionFailureReason.EFFECTIVE_PROVIDER_IDENTITY_UNAVAILABLE,
            detail="external_work configured adoption unsupported",
        )
    expected = expected_tenant_id
    if type(expected) is not str or not expected or expected != expected.strip():
        raise ExecutionIntegrationConfigurationAdoptionError(
            ExecutionIntegrationConfigurationAdoptionFailureReason.CONFIGURED_ADOPTION_TENANT_MISMATCH,
            detail="invalid expected tenant_id",
        )
    binding = adoption.configured_binding
    if binding.tenant_id != expected:
        raise ExecutionIntegrationConfigurationAdoptionError(
            ExecutionIntegrationConfigurationAdoptionFailureReason.CONFIGURED_ADOPTION_TENANT_MISMATCH,
            detail="binding tenant mismatch",
        )
    if binding.integration_category != adoption.integration_category:
        raise ExecutionIntegrationConfigurationAdoptionError(
            ExecutionIntegrationConfigurationAdoptionFailureReason.CONFIGURED_ADOPTION_CATEGORY_MISMATCH,
        )
    if adoption.integration_category != effective.integration_category:
        raise ExecutionIntegrationConfigurationAdoptionError(
            ExecutionIntegrationConfigurationAdoptionFailureReason.CONFIGURED_ADOPTION_CATEGORY_MISMATCH,
            detail="effective category mismatch",
        )
    if binding.provider_id != effective.provider_id:
        raise ExecutionIntegrationConfigurationAdoptionError(
            ExecutionIntegrationConfigurationAdoptionFailureReason.CONFIGURED_ADOPTION_PROVIDER_MISMATCH,
        )
    if binding.resource_scope != adoption.resource_scope:
        raise ExecutionIntegrationConfigurationAdoptionError(
            ExecutionIntegrationConfigurationAdoptionFailureReason.CONFIGURED_ADOPTION_RESOURCE_SCOPE_MISMATCH,
        )


__all__ = [
    "EffectiveIntegrationIdentity",
    "ExecutionIntegrationConfigurationAdoption",
    "ExecutionIntegrationConfigurationAdoptionError",
    "ExecutionIntegrationConfigurationAdoptionFailureReason",
    "IntegrationMaterializationKind",
    "validate_configured_adoption_match",
    "validate_configured_capability_binding_identity",
    "validate_effective_integration_identity",
    "validate_execution_integration_configuration_adoption",
]
