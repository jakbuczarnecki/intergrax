# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Requirement spine fact and commit port (TRACE-X-P5-R2-P4-R2)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    ExecutionId,
    RunId,
    TaskId,
    validate_event_id,
    validate_execution_id,
)
from intergrax.contracts.execution_integration_configuration_provenance import (
    ExecutionIntegrationConfigurationProvenanceMode,
    IntegrationConfigurationSubject,
    validate_integration_configuration_subject,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.runtime_event_type import RuntimeEventType
_REQUIREMENT_EVENT_KIND = RuntimeEventType.INTEGRATION_CONFIGURATION_PROVENANCE_REQUIREMENT_COMMITTED


@dataclass(frozen=True, slots=True)
class ExecutionIntegrationConfigurationProvenanceRequirementFact:
    tenant_id: str
    execution_id: ExecutionId
    subject: IntegrationConfigurationSubject
    mode: ExecutionIntegrationConfigurationProvenanceMode
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    factual_timestamp: datetime
    phase: ExecutionPhase
    severity: EventSeverity
    node_id: str | None = None
    agent_id: str | None = None
    step_id: str | None = None
    correlation_id: str | None = None
    parent_event_id: EventId | None = None
    traceparent: str | None = None
    tracestate: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.tenant_id) is not str or not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        validate_execution_id(self.execution_id)
        validate_integration_configuration_subject(self.subject)
        if self.factual_timestamp.tzinfo is None:
            raise ValueError("factual_timestamp must be timezone-aware")


class ExecutionIntegrationConfigurationProvenanceRequirementCommitStatus(StrEnum):
    COMMITTED = "committed"
    PERSISTENCE_UNAVAILABLE = "persistence_unavailable"


@dataclass(frozen=True, slots=True)
class ExecutionIntegrationConfigurationProvenanceRequirementCommitResult:
    status: ExecutionIntegrationConfigurationProvenanceRequirementCommitStatus
    event_id: EventId | None = None


@runtime_checkable
class ExecutionIntegrationConfigurationProvenanceRequirementCommitPort(Protocol):
    def commit_configured_adopted_requirement(
        self,
        fact: ExecutionIntegrationConfigurationProvenanceRequirementFact,
    ) -> ExecutionIntegrationConfigurationProvenanceRequirementCommitResult:
        ...


def derive_integration_configuration_provenance_requirement_event_id(
    *,
    tenant_id: str,
    execution_id: ExecutionId,
    subject: IntegrationConfigurationSubject,
) -> EventId:
    validate_integration_configuration_subject(subject)
    validate_execution_id(execution_id)
    material = {
        "event_kind": _REQUIREMENT_EVENT_KIND.value,
        "tenant_id": tenant_id,
        "execution_id": str(execution_id),
        "subject": {
            "integration_category": subject.integration_category.value,
            "provider_id": subject.provider_id,
            "resource_scope": subject.resource_scope,
            "configuration_type": subject.configuration_type,
        },
    }
    digest = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    ).hexdigest()[:32]
    return validate_event_id(f"evt_{digest}")


__all__ = [
    "ExecutionIntegrationConfigurationProvenanceRequirementCommitPort",
    "ExecutionIntegrationConfigurationProvenanceRequirementCommitResult",
    "ExecutionIntegrationConfigurationProvenanceRequirementCommitStatus",
    "ExecutionIntegrationConfigurationProvenanceRequirementFact",
    "derive_integration_configuration_provenance_requirement_event_id",
]
