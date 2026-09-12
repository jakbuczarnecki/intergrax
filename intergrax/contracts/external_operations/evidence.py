# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed diagnostic evidence for external operations (R1)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.external_operations.safety import (
    assert_no_secrets_in_audit_payload,
    sanitize_external_operation_text,
)

SCHEMA_EXTERNAL_OPERATION_EVIDENCE_V1: Final = "external_operation_evidence.v1"


class ExternalOperationEvidenceKind(StrEnum):
    PROVIDER_RESPONSE = "PROVIDER_RESPONSE"
    APPROVAL_RECORD = "APPROVAL_RECORD"
    EXECUTION_RESULT = "EXECUTION_RESULT"
    ADMISSION_DENIAL = "ADMISSION_DENIAL"
    PROVIDER_FAILURE = "PROVIDER_FAILURE"
    EXECUTION_FAILURE = "EXECUTION_FAILURE"


class ExternalOperationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    intent_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    kind: ExternalOperationEvidenceKind
    safe_summary: str = Field(min_length=1, max_length=512)
    recorded_at: datetime
    ref_artifacts: tuple[str, ...] = ()

    @field_validator("safe_summary")
    @classmethod
    def _sanitize_summary(cls, value: str) -> str:
        cleaned = sanitize_external_operation_text(value)
        assert_no_secrets_in_audit_payload((cleaned,))
        return cleaned


class ProviderFailureEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(min_length=1)
    failure_code: str = Field(min_length=1, max_length=128)
    safe_message: str = Field(min_length=1, max_length=512)

    @field_validator("safe_message")
    @classmethod
    def _sanitize(cls, value: str) -> str:
        return sanitize_external_operation_text(value)


class ExecutionFailureEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    failure_code: str = Field(min_length=1, max_length=128)
    safe_summary: str = Field(min_length=1, max_length=512)

    @field_validator("safe_summary")
    @classmethod
    def _sanitize(cls, value: str) -> str:
        return sanitize_external_operation_text(value)


class ExternalOperationFailed(BaseModel):
    """Structured failure — not exception-only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    attempt_id: str = Field(min_length=1)
    provider_failure: ProviderFailureEvidence | None = None
    execution_failure: ExecutionFailureEvidence | None = None

    @model_validator(mode="after")
    def _require_failure_payload(self) -> ExternalOperationFailed:
        if self.provider_failure is None and self.execution_failure is None:
            raise ValueError("ExternalOperationFailed requires provider or execution evidence")
        return self


class ProviderExecutionOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str = Field(pattern=r"^(success|failed|denied)$")
    safe_summary: str = Field(min_length=1, max_length=512)

    @field_validator("safe_summary")
    @classmethod
    def _sanitize(cls, value: str) -> str:
        return sanitize_external_operation_text(value)
