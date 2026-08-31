# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Typed functional validation evidence contract for observability export (DIAG-FUNCTIONAL-1)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_event_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.functional_evidence_bounds import (
    MAX_DIRECT_UPSTREAM_EVIDENCE_REFS,
)

PLATFORM_FUNCTIONAL_VALIDATION_EVIDENCE_SCHEMA = "platform_functional_validation_evidence.v1"


class FunctionalValidationOutcome(StrEnum):
    """Domain validator outcome — independent of execution terminal state."""

    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class FunctionalValidationKind(StrEnum):
    """Generic validation semantics — not pipeline- or scenario-specific."""

    ORACLE_ASSERTION = "oracle_assertion"
    DOMAIN_POLICY = "domain_policy"
    CONTRACT_ASSERTION = "contract_assertion"
    EXTERNAL_VALIDATOR = "external_validator"


class ExpectedActualRelation(StrEnum):
    """Bounded expected/actual relationship without raw payload content."""

    CONTAINS = "contains"
    EQUALS = "equals"
    SATISFIES = "satisfies"
    WITHIN_THRESHOLD = "within_threshold"
    SELECTED_MATCH = "selected_match"
    OTHER = "other"


class FunctionalValidatorRef(BaseModel):
    """Identity of the external/domain validator that produced the outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    validator_id: str
    validator_version: str = ""

    @field_validator("validator_id", "validator_version")
    @classmethod
    def _require_non_empty_identifier(cls, value: str) -> str:
        if type(value) is not str:
            raise TypeError("validator identifiers must be str")
        normalized = value.strip()
        if not normalized:
            raise ValueError("validator identifiers must be non-empty")
        if value != normalized:
            raise ValueError("validator identifiers must not contain leading or trailing whitespace")
        return normalized


class DiagnosticExecutionCorrelation(BaseModel):
    """Typed execution scope for functional validation and evidence correlation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId | None = None
    event_id: EventId | None = None

    @field_validator("task_id", mode="before")
    @classmethod
    def _validate_task_id_field(cls, value: object) -> TaskId:
        return validate_task_id(value)

    @field_validator("run_id", mode="before")
    @classmethod
    def _validate_run_id_field(cls, value: object) -> RunId:
        return validate_run_id(value)

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id_field(cls, value: object) -> AttemptId | None:
        if value is None:
            return None
        return validate_attempt_id(value)

    @field_validator("event_id", mode="before")
    @classmethod
    def _validate_event_id_field(cls, value: object) -> EventId | None:
        if value is None:
            return None
        return validate_event_id(value)

    @field_validator("tenant_id")
    @classmethod
    def _require_tenant_id(cls, value: str) -> str:
        if type(value) is not str:
            raise TypeError("tenant_id must be str")
        normalized = value.strip()
        if not normalized:
            raise ValueError("tenant_id must be non-empty and not whitespace-only")
        if value != normalized:
            raise ValueError("tenant_id must not contain leading or trailing whitespace")
        return normalized


class FunctionalValidationEvidence(BaseModel):
    """
    Bounded functional validation fact emitted by an external/domain validator.

    Does not alter execution terminal state. DIAG interprets this evidence.
    ``relation_summary`` is a bounded safe summary only — not canonical raw content.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["platform_functional_validation_evidence.v1"] = (
        PLATFORM_FUNCTIONAL_VALIDATION_EVIDENCE_SCHEMA
    )
    validation_id: EventId
    validator: FunctionalValidatorRef
    validation_kind: FunctionalValidationKind
    outcome: FunctionalValidationOutcome
    correlation: DiagnosticExecutionCorrelation
    expected_actual_relation: ExpectedActualRelation
    relation_summary: str = ""
    upstream_evidence_ids: tuple[EventId, ...] = ()
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("validation_id", mode="before")
    @classmethod
    def _validate_validation_id_field(cls, value: object) -> EventId:
        return validate_event_id(value)

    @field_validator("upstream_evidence_ids", mode="before")
    @classmethod
    def _validate_upstream_evidence_ids(
        cls,
        value: object,
    ) -> tuple[EventId, ...]:
        if value is None:
            return ()
        if type(value) is not tuple:
            raise TypeError("upstream_evidence_ids must be a tuple")
        normalized = tuple(validate_event_id(item) for item in value)
        if len(normalized) > MAX_DIRECT_UPSTREAM_EVIDENCE_REFS:
            raise ValueError(
                f"upstream_evidence_ids must contain <= {MAX_DIRECT_UPSTREAM_EVIDENCE_REFS} refs",
            )
        return normalized

    @field_validator("relation_summary")
    @classmethod
    def _validate_relation_summary(cls, value: str) -> str:
        if type(value) is not str:
            raise TypeError("relation_summary must be str")
        if len(value) > 512:
            raise ValueError("relation_summary must be <= 512 characters")
        return value

    @field_validator("recorded_at")
    @classmethod
    def _require_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("recorded_at must be timezone-aware")
        return value


__all__ = [
    "DiagnosticExecutionCorrelation",
    "ExpectedActualRelation",
    "FunctionalValidationEvidence",
    "FunctionalValidationKind",
    "FunctionalValidationOutcome",
    "FunctionalValidatorRef",
    "PLATFORM_FUNCTIONAL_VALIDATION_EVIDENCE_SCHEMA",
]
