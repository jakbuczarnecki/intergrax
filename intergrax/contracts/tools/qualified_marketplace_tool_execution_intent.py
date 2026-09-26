# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Durable Marketplace qualified Tool execution intent (S24-GAP-02-P3)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text

SCHEMA_QUALIFIED_MARKETPLACE_TOOL_EXECUTION_INTENT_V1: Final = (
    "qualified_marketplace_tool_execution_intent.v1"
)


class QualifiedMarketplaceToolExecutionIntentWriteOutcome(StrEnum):
    CREATED = "created"
    ALREADY_RECORDED_IDENTICAL = "already_recorded_identical"


class QualifiedMarketplaceToolExecutionIntentWriteResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: QualifiedMarketplaceToolExecutionIntentWriteOutcome


class QualifiedMarketplaceToolExecutionIntentError(Exception):
    """Base error for durable marketplace tool execution intent."""


class QualifiedMarketplaceToolExecutionIntentConflictError(
    QualifiedMarketplaceToolExecutionIntentError,
):
    """Same execution_request_id recorded with a different immutable payload."""


class QualifiedMarketplaceToolExecutionIntentUnavailableError(
    QualifiedMarketplaceToolExecutionIntentError,
):
    """Intent store backend unavailable."""


class QualifiedMarketplaceToolExecutionIntentIntegrityError(
    QualifiedMarketplaceToolExecutionIntentError,
):
    """Stored intent cannot be reconstructed into the canonical model."""


class QualifiedMarketplaceToolExecutionIntent(BaseModel):
    """Pre-EE durable execution intent — no business payload or authority scopes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["qualified_marketplace_tool_execution_intent.v1"] = (
        SCHEMA_QUALIFIED_MARKETPLACE_TOOL_EXECUTION_INTENT_V1
    )
    execution_request_id: str = Field(min_length=1)
    binding_operation_id: str = Field(min_length=1)
    resume_operation_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    worker_need_id: str = Field(min_length=1)
    qualified_subject_reference: str = Field(min_length=1)
    handoff_id: str = Field(min_length=1)
    selected_operation: str = Field(min_length=1)

    @field_validator(
        "execution_request_id",
        "binding_operation_id",
        "resume_operation_id",
        "tenant_id",
        "task_id",
        "worker_need_id",
        "qualified_subject_reference",
        "handoff_id",
        "selected_operation",
    )
    @classmethod
    def _validate_non_empty(cls, value: str, info: ValidationInfo) -> str:
        return require_non_empty_text(value, label=str(info.field_name))


@runtime_checkable
class QualifiedMarketplaceToolExecutionIntentRepository(Protocol):
    """SPI for durable pre-EE marketplace tool execution intent."""

    def record(
        self,
        intent: QualifiedMarketplaceToolExecutionIntent,
    ) -> QualifiedMarketplaceToolExecutionIntentWriteResult: ...

    def get(
        self,
        *,
        execution_request_id: str,
    ) -> QualifiedMarketplaceToolExecutionIntent | None: ...


__all__ = [
    "SCHEMA_QUALIFIED_MARKETPLACE_TOOL_EXECUTION_INTENT_V1",
    "QualifiedMarketplaceToolExecutionIntent",
    "QualifiedMarketplaceToolExecutionIntentConflictError",
    "QualifiedMarketplaceToolExecutionIntentIntegrityError",
    "QualifiedMarketplaceToolExecutionIntentRepository",
    "QualifiedMarketplaceToolExecutionIntentUnavailableError",
    "QualifiedMarketplaceToolExecutionIntentWriteOutcome",
    "QualifiedMarketplaceToolExecutionIntentWriteResult",
]
