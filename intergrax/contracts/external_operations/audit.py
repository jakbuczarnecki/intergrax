# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reconstructable audit chain for external operations (R1)."""

from __future__ import annotations

from datetime import datetime
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.external_operations.admission import OperationAdmissionDecision
from intergrax.contracts.external_operations.attempt import ExternalOperationAttemptLifecycle
from intergrax.contracts.external_operations.safety import (
    assert_no_secrets_in_audit_payload,
    sanitize_external_operation_text,
)

SCHEMA_EXTERNAL_OPERATION_AUDIT_V1: Final = "external_operation_audit.v1"


class ExternalOperationAuditRecord(BaseModel):
    """Operator-facing audit slice — no credentials."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    attempt_id: str = Field(min_length=1)
    intent_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    provider_id: str | None = Field(default=None, min_length=1)
    admission_decision: OperationAdmissionDecision
    execution_status: ExternalOperationAttemptLifecycle
    recorded_at: datetime
    actor: str = Field(min_length=1, max_length=256)
    evidence_refs: tuple[str, ...] = ()

    @field_validator("actor")
    @classmethod
    def _sanitize_actor(cls, value: str) -> str:
        return sanitize_external_operation_text(value)

    @field_validator("evidence_refs")
    @classmethod
    def _sanitize_refs(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        cleaned = tuple(sanitize_external_operation_text(ref) for ref in value)
        assert_no_secrets_in_audit_payload(cleaned)
        return cleaned
