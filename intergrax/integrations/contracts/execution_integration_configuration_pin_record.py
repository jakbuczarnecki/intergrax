# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed P2 pin record and requirement recovery staging (TRACE-X-P5-R2-P4-R2)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.execution_integration_configuration_provenance import (
    ExecutionIntegrationConfigurationProvenance,
    ExecutionIntegrationConfigurationProvenanceMode,
    IntegrationConfigurationSubject,
    validate_execution_integration_configuration_provenance,
    validate_integration_configuration_subject,
)


@dataclass(frozen=True, slots=True)
class ExecutionIntegrationConfigurationRequirementRecoveryStaging:
    """Immutable durable recovery staging bundled in the first pin row."""

    requirement_boundary_prepared_at: datetime
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    node_id: str | None = None
    agent_id: str | None = None
    step_id: str | None = None
    correlation_id: str | None = None
    traceparent: str | None = None
    tracestate: str | None = None

    def __post_init__(self) -> None:
        validate_requirement_recovery_staging(self)


@dataclass(frozen=True, slots=True)
class ExecutionIntegrationConfigurationPinRecord:
    subject: IntegrationConfigurationSubject
    provenance: ExecutionIntegrationConfigurationProvenance
    requirement_recovery_staging: ExecutionIntegrationConfigurationRequirementRecoveryStaging | None

    def __post_init__(self) -> None:
        validate_integration_configuration_subject(self.subject)
        validate_execution_integration_configuration_provenance(self.provenance)


def validate_requirement_recovery_staging(
    staging: ExecutionIntegrationConfigurationRequirementRecoveryStaging,
) -> None:
    prepared = staging.requirement_boundary_prepared_at
    if not isinstance(prepared, datetime):
        raise ValueError("requirement_boundary_prepared_at must be datetime")
    if prepared.tzinfo is None or prepared.utcoffset() is None:
        raise ValueError("requirement_boundary_prepared_at must be timezone-aware")
    validate_task_id(staging.task_id)
    validate_run_id(staging.run_id)
    validate_attempt_id(staging.attempt_id)


def find_pin_record_for_subject(
    records: tuple[ExecutionIntegrationConfigurationPinRecord, ...],
    subject: IntegrationConfigurationSubject,
) -> ExecutionIntegrationConfigurationPinRecord | None:
    validate_integration_configuration_subject(subject)
    for record in records:
        if record.subject == subject:
            return record
    return None


def obligation_requires_recovery_staging(
    provenance: ExecutionIntegrationConfigurationProvenance,
) -> bool:
    return (
        provenance.mode
        is ExecutionIntegrationConfigurationProvenanceMode.CONFIGURED_ADOPTED
    )


__all__ = [
    "ExecutionIntegrationConfigurationPinRecord",
    "ExecutionIntegrationConfigurationRequirementRecoveryStaging",
    "find_pin_record_for_subject",
    "obligation_requires_recovery_staging",
    "validate_requirement_recovery_staging",
]
