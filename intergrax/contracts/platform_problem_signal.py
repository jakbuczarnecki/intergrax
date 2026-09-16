# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vendor-neutral platform problem/error signal contract (OBS-PROBLEM-1)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.contracts.application_observability_attributes import (
    ApplicationObservabilityAttributes,
)
from intergrax.contracts.functional_validation_evidence import (
    FunctionalValidationEvidence,
    FunctionalValidationOutcome,
)
from intergrax.contracts.observability_artifact_reference import (
    ObservabilityArtifactReference,
)

PLATFORM_PROBLEM_SIGNAL_SCHEMA = "platform_problem_signal.v1"

PROBLEM_SEVERITY_INFO = "info"
PROBLEM_SEVERITY_WARNING = "warning"
PROBLEM_SEVERITY_ERROR = "error"
PROBLEM_SEVERITY_CRITICAL = "critical"

PROBLEM_STATUS_DETECTED = "detected"
PROBLEM_STATUS_RESOLVED = "resolved"
PROBLEM_STATUS_IGNORED = "ignored"

PROBLEM_KIND_PLATFORM_EXCEPTION = "platform.exception"
PROBLEM_KIND_PLATFORM_CONFIGURATION_ERROR = "platform.configuration_error"
PROBLEM_KIND_PLATFORM_POLICY_VIOLATION = "platform.policy_violation"
PROBLEM_KIND_PLATFORM_TOOL_FAILURE = "platform.tool_failure"
PROBLEM_KIND_PLATFORM_RAG_FAILURE = "platform.rag_failure"
PROBLEM_KIND_PLATFORM_ARTIFACT_FAILURE = "platform.artifact_failure"
PROBLEM_KIND_PLATFORM_INTEGRATION_FAILURE = "platform.integration_failure"
PROBLEM_KIND_PLATFORM_OBSERVABILITY_EXPORT_FAILURE = (
    "platform.observability_export_failure"
)
PROBLEM_KIND_PLATFORM_UNEXPECTED_STATE = "platform.unexpected_state"
PROBLEM_KIND_PLATFORM_APPLICATION_FAILURE = "platform.application_failure"
PROBLEM_KIND_PLATFORM_FUNCTIONAL_OUTCOME_INVALID = "platform.functional_outcome_invalid"
PROBLEM_KIND_PLATFORM_EXTERNAL_EFFECT_RELIABILITY = (
    "platform.external_effect_reliability"
)

PROBLEM_SOURCE_LAYER_VALIDATION = "validation"

PROBLEM_SOURCE_LAYER_RUNTIME = "runtime"
PROBLEM_SOURCE_LAYER_AGENT = "agent"
PROBLEM_SOURCE_LAYER_APPLICATION = "application"
PROBLEM_SOURCE_LAYER_INTEGRATION = "integration"
PROBLEM_SOURCE_LAYER_OBSERVABILITY = "observability"
PROBLEM_SOURCE_LAYER_POLICY = "policy"
PROBLEM_SOURCE_LAYER_TOOL = "tool"
PROBLEM_SOURCE_LAYER_RAG = "rag"


class PlatformProblemSignal(BaseModel):
    """Vendor-neutral platform problem/error signal with plugin-extensible taxonomy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["platform_problem_signal.v1"] = (
        PLATFORM_PROBLEM_SIGNAL_SCHEMA
    )

    problem_id: str = ""
    problem_kind: str
    severity: str = PROBLEM_SEVERITY_ERROR
    source_layer: str = ""
    source_component: str = ""
    status: str = PROBLEM_STATUS_DETECTED

    safe_message: str = ""
    error_code: str = ""
    exception_type: str | None = None

    run_id: str = ""
    task_id: str = ""
    event_id: str = ""
    agent_id: str = ""
    tool_id: str = ""
    capability: str = ""
    correlation_id: str = ""
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    application_attributes: ApplicationObservabilityAttributes | None = None
    agent_attributes: ApplicationObservabilityAttributes | None = None
    artifact_refs: tuple[ObservabilityArtifactReference, ...] = ()
    functional_validation: FunctionalValidationEvidence | None = None

    @model_validator(mode="after")
    def _validate_functional_validation_invariant(self) -> PlatformProblemSignal:
        validation = self.functional_validation
        if validation is not None:
            if self.problem_kind != PROBLEM_KIND_PLATFORM_FUNCTIONAL_OUTCOME_INVALID:
                raise ValueError(
                    "functional_validation requires problem_kind platform.functional_outcome_invalid",
                )
            if validation.outcome is not FunctionalValidationOutcome.FAILED:
                raise ValueError(
                    "functional_validation requires FAILED outcome for functional outcome invalid signal",
                )
            correlation = validation.correlation
            if self.task_id and self.task_id != str(correlation.task_id):
                raise ValueError("task_id must match functional_validation correlation")
            if self.run_id and self.run_id != str(correlation.run_id):
                raise ValueError("run_id must match functional_validation correlation")
        elif self.problem_kind == PROBLEM_KIND_PLATFORM_FUNCTIONAL_OUTCOME_INVALID:
            raise ValueError(
                "platform.functional_outcome_invalid requires functional_validation evidence",
            )
        return self


__all__ = [
    "PLATFORM_PROBLEM_SIGNAL_SCHEMA",
    "PROBLEM_KIND_PLATFORM_APPLICATION_FAILURE",
    "PROBLEM_KIND_PLATFORM_ARTIFACT_FAILURE",
    "PROBLEM_KIND_PLATFORM_CONFIGURATION_ERROR",
    "PROBLEM_KIND_PLATFORM_EXCEPTION",
    "PROBLEM_KIND_PLATFORM_EXTERNAL_EFFECT_RELIABILITY",
    "PROBLEM_KIND_PLATFORM_FUNCTIONAL_OUTCOME_INVALID",
    "PROBLEM_KIND_PLATFORM_INTEGRATION_FAILURE",
    "PROBLEM_KIND_PLATFORM_OBSERVABILITY_EXPORT_FAILURE",
    "PROBLEM_KIND_PLATFORM_POLICY_VIOLATION",
    "PROBLEM_KIND_PLATFORM_RAG_FAILURE",
    "PROBLEM_KIND_PLATFORM_TOOL_FAILURE",
    "PROBLEM_KIND_PLATFORM_UNEXPECTED_STATE",
    "PROBLEM_SEVERITY_CRITICAL",
    "PROBLEM_SEVERITY_ERROR",
    "PROBLEM_SEVERITY_INFO",
    "PROBLEM_SEVERITY_WARNING",
    "PROBLEM_SOURCE_LAYER_AGENT",
    "PROBLEM_SOURCE_LAYER_APPLICATION",
    "PROBLEM_SOURCE_LAYER_INTEGRATION",
    "PROBLEM_SOURCE_LAYER_OBSERVABILITY",
    "PROBLEM_SOURCE_LAYER_POLICY",
    "PROBLEM_SOURCE_LAYER_RAG",
    "PROBLEM_SOURCE_LAYER_RUNTIME",
    "PROBLEM_SOURCE_LAYER_TOOL",
    "PROBLEM_SOURCE_LAYER_VALIDATION",
    "PROBLEM_STATUS_DETECTED",
    "PROBLEM_STATUS_IGNORED",
    "PROBLEM_STATUS_RESOLVED",
    "PlatformProblemSignal",
]
