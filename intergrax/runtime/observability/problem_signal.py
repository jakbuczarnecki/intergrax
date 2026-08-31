# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vendor-neutral platform problem/error signal contract (OBS-PROBLEM-1)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from intergrax.runtime.observability.functional_validation_evidence import FunctionalValidationEvidence
from intergrax.runtime.observability.export_attributes import (
    ApplicationObservabilityAttributes,
    ObservabilityArtifactReference,
)
from intergrax.runtime.observability.export_boundary import FORBIDDEN_EXPORT_CONTENT_FIELDS

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
PROBLEM_KIND_PLATFORM_OBSERVABILITY_EXPORT_FAILURE = "platform.observability_export_failure"
PROBLEM_KIND_PLATFORM_UNEXPECTED_STATE = "platform.unexpected_state"
PROBLEM_KIND_PLATFORM_APPLICATION_FAILURE = "platform.application_failure"
PROBLEM_KIND_PLATFORM_FUNCTIONAL_OUTCOME_INVALID = "platform.functional_outcome_invalid"

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

    schema_version: Literal["platform_problem_signal.v1"] = PLATFORM_PROBLEM_SIGNAL_SCHEMA

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


def _problem_signal_json_is_content_safe(serialized: str) -> bool:
    """Return False when serialized JSON exposes forbidden raw-content field names."""
    for key in FORBIDDEN_EXPORT_CONTENT_FIELDS:
        if f'"{key}"' in serialized:
            return False
    return True


def problem_signal_is_content_safe(signal: PlatformProblemSignal) -> bool:
    """Return False when serialized problem signal exposes forbidden raw-content field names."""
    return _problem_signal_json_is_content_safe(signal.model_dump_json())
