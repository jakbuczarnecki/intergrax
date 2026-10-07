# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Configuration opportunity contracts — facts, final opportunity, read port, provider SPI (TRACE-X-P5-R2-P1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NewType, Protocol, runtime_checkable

from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRisk
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.existing_capability_configuration import (
    IntegrationConfigurationPayload,
)

ConfigurationOpportunityRef = NewType("ConfigurationOpportunityRef", str)


class ExistingCapabilityConfigurationOpportunityLookupFailureReason(StrEnum):
    NOT_FOUND = "NOT_FOUND"
    TENANT_MISMATCH = "TENANT_MISMATCH"
    STALE = "STALE"
    AMBIGUOUS = "AMBIGUOUS"
    INVALID = "INVALID"
    FINGERPRINT_MISMATCH = "FINGERPRINT_MISMATCH"


class ExistingCapabilityConfigurationOpportunityLookupError(Exception):
    """Typed opportunity lookup failure — reason is the semantic API."""

    def __init__(
        self,
        reason: ExistingCapabilityConfigurationOpportunityLookupFailureReason,
        *,
        detail: str = "",
    ) -> None:
        self.reason = reason
        self.detail = detail
        message = reason.value if not detail else f"{reason.value}: {detail}"
        super().__init__(message)


def validate_configuration_opportunity_ref(value: object) -> ConfigurationOpportunityRef:
    if type(value) is not str:
        raise TypeError(
            f"ConfigurationOpportunityRef must be str, got {type(value).__name__}"
        )
    if not value or not value.strip():
        raise ValueError("ConfigurationOpportunityRef must be non-empty")
    if value != value.strip():
        raise ValueError(
            "ConfigurationOpportunityRef must not contain leading or trailing whitespace"
        )
    return ConfigurationOpportunityRef(value)


@dataclass(frozen=True, slots=True)
class ExistingCapabilityConfigurationOpportunityFacts:
    """Provider/configuration facts — no permission, risk authority, or execution identity."""

    tenant_id: str
    integration_category: IntegrationCategory
    provider_id: str
    resource_scope: str
    current_revision: str
    configuration: IntegrationConfigurationPayload
    configuration_fingerprint: str

    def __post_init__(self) -> None:
        validate_existing_capability_configuration_opportunity_facts(self)


@dataclass(frozen=True, slots=True)
class ExistingCapabilityConfigurationOpportunity:
    """Immutable opportunity — configuration could be realized; not configured or effective."""

    configuration_ref: ConfigurationOpportunityRef
    tenant_id: str
    integration_category: IntegrationCategory
    provider_id: str
    resource_scope: str
    current_revision: str
    configuration: IntegrationConfigurationPayload
    configuration_fingerprint: str
    risk_classification: ControlPlaneMutationRisk

    def __post_init__(self) -> None:
        validate_existing_capability_configuration_opportunity(self)


def validate_existing_capability_configuration_opportunity_facts(
    facts: ExistingCapabilityConfigurationOpportunityFacts,
) -> None:
    category = facts.integration_category
    if not isinstance(category, IntegrationCategory):
        raise ExistingCapabilityConfigurationOpportunityLookupError(
            ExistingCapabilityConfigurationOpportunityLookupFailureReason.INVALID,
            detail="invalid integration_category",
        )
    tenant = facts.tenant_id
    if type(tenant) is not str or not tenant or tenant != tenant.strip():
        raise ExistingCapabilityConfigurationOpportunityLookupError(
            ExistingCapabilityConfigurationOpportunityLookupFailureReason.INVALID,
            detail="invalid tenant_id",
        )
    provider = facts.provider_id
    if type(provider) is not str or not provider or provider != provider.strip():
        raise ExistingCapabilityConfigurationOpportunityLookupError(
            ExistingCapabilityConfigurationOpportunityLookupFailureReason.INVALID,
            detail="invalid provider_id",
        )
    scope = facts.resource_scope
    if type(scope) is not str or not scope or scope != scope.strip():
        raise ExistingCapabilityConfigurationOpportunityLookupError(
            ExistingCapabilityConfigurationOpportunityLookupFailureReason.INVALID,
            detail="invalid resource_scope",
        )
    revision = facts.current_revision
    if type(revision) is not str or not revision or revision != revision.strip():
        raise ExistingCapabilityConfigurationOpportunityLookupError(
            ExistingCapabilityConfigurationOpportunityLookupFailureReason.INVALID,
            detail="invalid current_revision",
        )
    _validate_configuration_payload_identity(facts.configuration)
    fingerprint = facts.configuration_fingerprint
    if type(fingerprint) is not str or not fingerprint or fingerprint != fingerprint.strip():
        raise ExistingCapabilityConfigurationOpportunityLookupError(
            ExistingCapabilityConfigurationOpportunityLookupFailureReason.INVALID,
            detail="invalid configuration_fingerprint",
        )
    if fingerprint != facts.configuration.configuration_fingerprint.strip():
        raise ExistingCapabilityConfigurationOpportunityLookupError(
            ExistingCapabilityConfigurationOpportunityLookupFailureReason.FINGERPRINT_MISMATCH,
            detail="facts fingerprint mismatch",
        )


def validate_existing_capability_configuration_opportunity(
    opportunity: ExistingCapabilityConfigurationOpportunity,
) -> None:
    validate_configuration_opportunity_ref(opportunity.configuration_ref)
    category = opportunity.integration_category
    if not isinstance(category, IntegrationCategory):
        raise ExistingCapabilityConfigurationOpportunityLookupError(
            ExistingCapabilityConfigurationOpportunityLookupFailureReason.INVALID,
            detail="invalid integration_category",
        )
    risk = opportunity.risk_classification
    if not isinstance(risk, ControlPlaneMutationRisk):
        raise ExistingCapabilityConfigurationOpportunityLookupError(
            ExistingCapabilityConfigurationOpportunityLookupFailureReason.INVALID,
            detail="invalid risk_classification",
        )
    tenant = opportunity.tenant_id
    if type(tenant) is not str or not tenant or tenant != tenant.strip():
        raise ExistingCapabilityConfigurationOpportunityLookupError(
            ExistingCapabilityConfigurationOpportunityLookupFailureReason.INVALID,
            detail="invalid tenant_id",
        )
    provider = opportunity.provider_id
    if type(provider) is not str or not provider or provider != provider.strip():
        raise ExistingCapabilityConfigurationOpportunityLookupError(
            ExistingCapabilityConfigurationOpportunityLookupFailureReason.INVALID,
            detail="invalid provider_id",
        )
    scope = opportunity.resource_scope
    if type(scope) is not str or not scope or scope != scope.strip():
        raise ExistingCapabilityConfigurationOpportunityLookupError(
            ExistingCapabilityConfigurationOpportunityLookupFailureReason.INVALID,
            detail="invalid resource_scope",
        )
    revision = opportunity.current_revision
    if type(revision) is not str or not revision or revision != revision.strip():
        raise ExistingCapabilityConfigurationOpportunityLookupError(
            ExistingCapabilityConfigurationOpportunityLookupFailureReason.INVALID,
            detail="invalid current_revision",
        )
    _validate_configuration_payload_identity(opportunity.configuration)
    fingerprint = opportunity.configuration_fingerprint
    if type(fingerprint) is not str or not fingerprint or fingerprint != fingerprint.strip():
        raise ExistingCapabilityConfigurationOpportunityLookupError(
            ExistingCapabilityConfigurationOpportunityLookupFailureReason.INVALID,
            detail="invalid configuration_fingerprint",
        )
    if fingerprint != opportunity.configuration.configuration_fingerprint.strip():
        raise ExistingCapabilityConfigurationOpportunityLookupError(
            ExistingCapabilityConfigurationOpportunityLookupFailureReason.FINGERPRINT_MISMATCH,
            detail="opportunity fingerprint mismatch",
        )


def _validate_configuration_payload_identity(
    configuration: IntegrationConfigurationPayload,
) -> None:
    config_type = configuration.configuration_type
    if type(config_type) is not str or not config_type or config_type != config_type.strip():
        raise ExistingCapabilityConfigurationOpportunityLookupError(
            ExistingCapabilityConfigurationOpportunityLookupFailureReason.INVALID,
            detail="missing configuration_type",
        )
    config_version = configuration.configuration_version
    if (
        type(config_version) is not str
        or not config_version
        or config_version != config_version.strip()
    ):
        raise ExistingCapabilityConfigurationOpportunityLookupError(
            ExistingCapabilityConfigurationOpportunityLookupFailureReason.INVALID,
            detail="missing configuration_version",
        )


@runtime_checkable
class ExistingCapabilityConfigurationOpportunityReadPort(Protocol):
    """Exact tenant + opaque ref → exact opportunity — no latest or best-effort semantics."""

    def read_exact(
        self,
        *,
        tenant_id: str,
        configuration_ref: ConfigurationOpportunityRef,
    ) -> ExistingCapabilityConfigurationOpportunity:
        ...


@runtime_checkable
class ExistingCapabilityConfigurationOpportunityProvider(Protocol):
    """Plugin supplies factual candidates — Integrations owns final opportunity and risk."""

    def discover_opportunity_facts(
        self,
        *,
        tenant_id: str,
    ) -> tuple[ExistingCapabilityConfigurationOpportunityFacts, ...]:
        ...


@runtime_checkable
class ExistingCapabilityConfigurationMutationRiskPolicy(Protocol):
    """Domain risk classification only — not Governance permission or provider materialization."""

    def classify(
        self,
        facts: ExistingCapabilityConfigurationOpportunityFacts,
    ) -> ControlPlaneMutationRisk:
        ...


__all__ = [
    "ConfigurationOpportunityRef",
    "ExistingCapabilityConfigurationMutationRiskPolicy",
    "ExistingCapabilityConfigurationOpportunity",
    "ExistingCapabilityConfigurationOpportunityFacts",
    "ExistingCapabilityConfigurationOpportunityLookupError",
    "ExistingCapabilityConfigurationOpportunityLookupFailureReason",
    "ExistingCapabilityConfigurationOpportunityProvider",
    "ExistingCapabilityConfigurationOpportunityReadPort",
    "validate_configuration_opportunity_ref",
    "validate_existing_capability_configuration_opportunity",
    "validate_existing_capability_configuration_opportunity_facts",
]
