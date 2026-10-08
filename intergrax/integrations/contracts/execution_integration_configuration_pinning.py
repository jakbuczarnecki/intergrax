# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Integrations-owned execution integration configuration pinning (TRACE-X-P5-R2-P2)."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_identity import ExecutionId
from intergrax.contracts.execution_integration_configuration_provenance import (
    ExecutionIntegrationConfigurationProvenance,
    ExecutionIntegrationConfigurationProvenanceMode,
    IntegrationConfigurationSubject,
    validate_execution_integration_configuration_provenance,
    validate_integration_configuration_subject,
)
from intergrax.integrations.contracts.base import IntegrationCategory


class ExecutionIntegrationConfigurationPinningFailureReason(StrEnum):
    CONFLICT = "CONFLICT"
    NOT_FOUND = "NOT_FOUND"
    CORRUPT_RECORD = "CORRUPT_RECORD"
    UNSUPPORTED_SCHEMA_VERSION = "UNSUPPORTED_SCHEMA_VERSION"
    TENANT_MISMATCH = "TENANT_MISMATCH"
    SUBJECT_MISMATCH = "SUBJECT_MISMATCH"
    INVALID = "INVALID"


class ExecutionIntegrationConfigurationPinningError(Exception):
    """Typed pinning persistence failure — fail closed."""

    def __init__(
        self,
        reason: ExecutionIntegrationConfigurationPinningFailureReason,
        *,
        detail: str = "",
    ) -> None:
        self.reason = reason
        self.detail = detail
        message = reason.value if not detail else f"{reason.value}: {detail}"
        super().__init__(message)


def validate_pin_subject_against_provenance(
    *,
    subject: IntegrationConfigurationSubject,
    provenance: ExecutionIntegrationConfigurationProvenance,
) -> None:
    """Enforce subject/provenance consistency before durable pin."""
    validate_integration_configuration_subject(subject)
    validate_execution_integration_configuration_provenance(provenance)
    effective = provenance.effective
    category = subject.integration_category
    if not isinstance(category, IntegrationCategory):
        raise ExecutionIntegrationConfigurationPinningError(
            ExecutionIntegrationConfigurationPinningFailureReason.INVALID,
            detail="invalid subject integration_category",
        )
    if category != effective.integration_category:
        raise ExecutionIntegrationConfigurationPinningError(
            ExecutionIntegrationConfigurationPinningFailureReason.SUBJECT_MISMATCH,
            detail="subject category does not match effective identity",
        )
    if subject.provider_id != effective.provider_id:
        raise ExecutionIntegrationConfigurationPinningError(
            ExecutionIntegrationConfigurationPinningFailureReason.SUBJECT_MISMATCH,
            detail="subject provider does not match effective identity",
        )
    mode = provenance.mode
    configured = provenance.configured
    if mode == ExecutionIntegrationConfigurationProvenanceMode.CONFIGURED_ADOPTED:
        if configured is None:
            raise ExecutionIntegrationConfigurationPinningError(
                ExecutionIntegrationConfigurationPinningFailureReason.INVALID,
                detail="CONFIGURED_ADOPTED requires configured slice",
            )
        if subject.integration_category != configured.integration_category:
            raise ExecutionIntegrationConfigurationPinningError(
                ExecutionIntegrationConfigurationPinningFailureReason.SUBJECT_MISMATCH,
                detail="subject category does not match configured slice",
            )
        if subject.provider_id != configured.provider_id:
            raise ExecutionIntegrationConfigurationPinningError(
                ExecutionIntegrationConfigurationPinningFailureReason.SUBJECT_MISMATCH,
                detail="subject provider does not match configured slice",
            )
        if subject.resource_scope != configured.resource_scope:
            raise ExecutionIntegrationConfigurationPinningError(
                ExecutionIntegrationConfigurationPinningFailureReason.SUBJECT_MISMATCH,
                detail="subject resource_scope does not match configured slice",
            )
        if subject.configuration_type != configured.configuration_type:
            raise ExecutionIntegrationConfigurationPinningError(
                ExecutionIntegrationConfigurationPinningFailureReason.SUBJECT_MISMATCH,
                detail="subject configuration_type does not match configured slice",
            )
    elif mode == ExecutionIntegrationConfigurationProvenanceMode.EFFECTIVE_ONLY:
        if configured is not None:
            raise ExecutionIntegrationConfigurationPinningError(
                ExecutionIntegrationConfigurationPinningFailureReason.INVALID,
                detail="EFFECTIVE_ONLY forbids configured slice",
            )


@runtime_checkable
class ExecutionIntegrationConfigurationPinningStore(Protocol):
    """Durable immutable provenance pin — tenant + ExecutionId + subject identity."""

    def pin(
        self,
        *,
        subject: IntegrationConfigurationSubject,
        provenance: ExecutionIntegrationConfigurationProvenance,
    ) -> None:
        ...

    def read_all(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
    ) -> tuple[ExecutionIntegrationConfigurationProvenance, ...]:
        ...


__all__ = [
    "ExecutionIntegrationConfigurationPinningError",
    "ExecutionIntegrationConfigurationPinningFailureReason",
    "ExecutionIntegrationConfigurationPinningStore",
    "validate_pin_subject_against_provenance",
]
