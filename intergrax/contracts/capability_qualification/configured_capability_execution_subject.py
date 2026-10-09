# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed CONFIGURE_EXISTING execution subject — post-adoption, pre-binding."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey

SCHEMA_CONFIGURED_CAPABILITY_EXECUTION_SUBJECT_V1: Final = (
    "configured_capability_execution_subject.v1"
)


def derive_configured_capability_execution_subject_reference(
    *,
    recovery_decision_id: str,
    decision_id: str,
    configuration_fingerprint: str,
) -> str:
    recovery = require_non_empty_text(
        recovery_decision_id,
        label="recovery_decision_id",
    )
    decision = require_non_empty_text(decision_id, label="decision_id")
    fingerprint = require_non_empty_text(
        configuration_fingerprint,
        label="configuration_fingerprint",
    )
    return (
        f"configured-capability-execution:{recovery}:{decision}:{fingerprint}"
    )


def derive_configured_capability_execution_operation_id(
    *,
    recovery_decision_id: str,
    decision_id: str,
) -> str:
    recovery = require_non_empty_text(
        recovery_decision_id,
        label="recovery_decision_id",
    )
    decision = require_non_empty_text(decision_id, label="decision_id")
    return f"configured-capability-execution:{recovery}:{decision}"


def derive_configured_capability_binding_operation_id(
    *,
    configured_execution_operation_id: str,
    subject_reference: str,
) -> str:
    execution_op = require_non_empty_text(
        configured_execution_operation_id,
        label="configured_execution_operation_id",
    )
    subject = require_non_empty_text(subject_reference, label="subject_reference")
    return f"configured-capability-binding:{execution_op}:{subject}"


def derive_configuration_adoption_identity(
    *,
    recovery_decision_id: str,
    decision_id: str,
    configuration_fingerprint: str,
) -> str:
    return (
        f"execution-integration-configuration-adoption:"
        f"{derive_configured_capability_execution_operation_id(recovery_decision_id=recovery_decision_id, decision_id=decision_id)}:"
        f"{require_non_empty_text(configuration_fingerprint, label='configuration_fingerprint')}"
    )


class ConfiguredCapabilityExecutionSubject(BaseModel):
    """Immutable configured execution anchor — no provider objects or governance decisions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["configured_capability_execution_subject.v1"] = (
        SCHEMA_CONFIGURED_CAPABILITY_EXECUTION_SUBJECT_V1
    )
    tenant_id: str = Field(min_length=1)
    worker_need_id: str = Field(min_length=1)
    recovery_decision_id: str = Field(min_length=1)
    decision_id: str = Field(min_length=1)
    capability_identity: CapabilityIdentityKey
    configuration_adoption_identity: str = Field(min_length=1)
    configuration_fingerprint: str = Field(min_length=1)
    selected_operations: tuple[str, ...] = Field(min_length=1)

    @field_validator(
        "tenant_id",
        "worker_need_id",
        "recovery_decision_id",
        "decision_id",
        "configuration_adoption_identity",
        "configuration_fingerprint",
    )
    @classmethod
    def _validate_non_empty(cls, value: str, info: ValidationInfo) -> str:
        return require_non_empty_text(value, label=str(info.field_name))

    @field_validator("selected_operations")
    @classmethod
    def _validate_operations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("selected_operations must be non-empty")
        return tuple(
            require_non_empty_text(item, label="selected_operations") for item in value
        )

    @property
    def subject_reference(self) -> str:
        return derive_configured_capability_execution_subject_reference(
            recovery_decision_id=self.recovery_decision_id,
            decision_id=self.decision_id,
            configuration_fingerprint=self.configuration_fingerprint,
        )


__all__ = [
    "ConfiguredCapabilityExecutionSubject",
    "SCHEMA_CONFIGURED_CAPABILITY_EXECUTION_SUBJECT_V1",
    "derive_configured_capability_binding_operation_id",
    "derive_configured_capability_execution_operation_id",
    "derive_configured_capability_execution_subject_reference",
    "derive_configuration_adoption_identity",
]
