# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neutral execution integration configuration provenance (TRACE-X-P5-R2-P1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.execution_integration_configuration import (
    EffectiveIntegrationIdentity,
)


@dataclass(frozen=True, slots=True, order=True)
class IntegrationConfigurationSubject:
    """Configured-adoption subject for later persistence — not fabricated for EFFECTIVE_ONLY."""

    integration_category: IntegrationCategory
    provider_id: str
    resource_scope: str
    configuration_type: str

    def __post_init__(self) -> None:
        validate_integration_configuration_subject(self)


@dataclass(frozen=True, slots=True)
class ConfiguredIntegrationProvenanceSlice:
    """Factual configured projection — not a second configuration authority."""

    tenant_id: str
    integration_category: IntegrationCategory
    provider_id: str
    resource_scope: str
    configuration_type: str
    configuration_version: str
    configuration_fingerprint: str
    realization_evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_configured_integration_provenance_slice(self)


class ExecutionIntegrationConfigurationProvenanceMode(StrEnum):
    CONFIGURED_ADOPTED = "configured_adopted"
    EFFECTIVE_ONLY = "effective_only"


class ExecutionIntegrationConfigurationProvenanceReadStatus(StrEnum):
    NOT_CONFIGURED = "not_configured"
    CONFIGURED = "configured"
    REQUIRED_MISSING = "required_missing"


@dataclass(frozen=True, slots=True)
class ExecutionIntegrationConfigurationProvenance:
    tenant_id: str
    execution_id: ExecutionId
    mode: ExecutionIntegrationConfigurationProvenanceMode
    effective: EffectiveIntegrationIdentity
    configured: ConfiguredIntegrationProvenanceSlice | None

    def __post_init__(self) -> None:
        validate_execution_integration_configuration_provenance(self)


@runtime_checkable
class ExecutionIntegrationConfigurationProvenanceReader(Protocol):
    """Read-only execution-scoped provenance — no pin, latest, or write semantics."""

    def read_all(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
    ) -> tuple[ExecutionIntegrationConfigurationProvenance, ...]:
        ...


def require_tenant_id_for_integration_configuration_provenance(tenant_id: str) -> str:
    if type(tenant_id) is not str or not tenant_id or tenant_id != tenant_id.strip():
        raise ValueError("tenant_id is required for integration configuration provenance")
    return tenant_id


def validate_integration_configuration_subject(
    subject: IntegrationConfigurationSubject,
) -> None:
    provider = subject.provider_id
    if type(provider) is not str or not provider or provider != provider.strip():
        raise ValueError("integration configuration subject provider_id invalid")
    scope = subject.resource_scope
    if type(scope) is not str or not scope or scope != scope.strip():
        raise ValueError("integration configuration subject resource_scope invalid")
    config_type = subject.configuration_type
    if type(config_type) is not str or not config_type or config_type != config_type.strip():
        raise ValueError("integration configuration subject configuration_type invalid")


def validate_configured_integration_provenance_slice(
    slice_: ConfiguredIntegrationProvenanceSlice,
) -> None:
    tenant = slice_.tenant_id
    if type(tenant) is not str or not tenant or tenant != tenant.strip():
        raise ValueError("configured provenance tenant_id invalid")
    provider = slice_.provider_id
    if type(provider) is not str or not provider or provider != provider.strip():
        raise ValueError("configured provenance provider_id invalid")
    scope = slice_.resource_scope
    if type(scope) is not str or not scope or scope != scope.strip():
        raise ValueError("configured provenance resource_scope invalid")
    config_type = slice_.configuration_type
    if type(config_type) is not str or not config_type or config_type != config_type.strip():
        raise ValueError("configured provenance configuration_type invalid")
    config_version = slice_.configuration_version
    if (
        type(config_version) is not str
        or not config_version
        or config_version != config_version.strip()
    ):
        raise ValueError("configured provenance configuration_version invalid")
    fingerprint = slice_.configuration_fingerprint
    if type(fingerprint) is not str or not fingerprint or fingerprint != fingerprint.strip():
        raise ValueError("configured provenance configuration_fingerprint invalid")


def validate_execution_integration_configuration_provenance(
    provenance: ExecutionIntegrationConfigurationProvenance,
) -> None:
    require_tenant_id_for_integration_configuration_provenance(provenance.tenant_id)
    validate_execution_id(provenance.execution_id)
    mode = provenance.mode
    configured = provenance.configured
    effective = provenance.effective
    if mode == ExecutionIntegrationConfigurationProvenanceMode.CONFIGURED_ADOPTED:
        if configured is None:
            raise ValueError("CONFIGURED_ADOPTED requires configured slice")
        validate_configured_integration_provenance_slice(configured)
        if configured.tenant_id != provenance.tenant_id:
            raise ValueError("configured provenance tenant mismatch")
        if configured.integration_category != effective.integration_category:
            raise ValueError("configured provenance category mismatch")
        if configured.provider_id != effective.provider_id:
            raise ValueError("configured provenance provider mismatch")
    elif mode == ExecutionIntegrationConfigurationProvenanceMode.EFFECTIVE_ONLY:
        if configured is not None:
            raise ValueError("EFFECTIVE_ONLY forbids configured slice")
    else:
        raise ValueError("unknown provenance mode")


def validate_execution_integration_configuration_provenance_record(
    record: ExecutionIntegrationConfigurationProvenance,
    *,
    expected_tenant_id: str,
    expected_execution_id: ExecutionId,
) -> None:
    tenant = require_tenant_id_for_integration_configuration_provenance(expected_tenant_id)
    if record.tenant_id != tenant:
        raise ValueError("integration configuration provenance tenant mismatch")
    if validate_execution_id(record.execution_id) != validate_execution_id(
        expected_execution_id
    ):
        raise ValueError("integration configuration provenance execution_id mismatch")
    validate_execution_integration_configuration_provenance(record)


__all__ = [
    "ConfiguredIntegrationProvenanceSlice",
    "ExecutionIntegrationConfigurationProvenance",
    "ExecutionIntegrationConfigurationProvenanceMode",
    "ExecutionIntegrationConfigurationProvenanceReadStatus",
    "ExecutionIntegrationConfigurationProvenanceReader",
    "IntegrationConfigurationSubject",
    "require_tenant_id_for_integration_configuration_provenance",
    "validate_configured_integration_provenance_slice",
    "validate_execution_integration_configuration_provenance",
    "validate_execution_integration_configuration_provenance_record",
    "validate_integration_configuration_subject",
]
