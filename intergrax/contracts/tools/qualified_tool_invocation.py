# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed qualified Tool business invocation material and resolver contracts (S24-GAP-02-P3)."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
)
from intergrax.contracts.execution_identity import TaskId


class QualifiedToolInvocationMaterialRequest(BaseModel):
    """Application-owned business material request — no runtime Task dependency."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    execution_request_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    task_id: TaskId
    selected_operation: str = Field(min_length=1)
    qualified_subject_reference: str = Field(min_length=1)
    handoff_id: str = Field(min_length=1)
    worker_need_id: str = Field(min_length=1)
    activated_tool_id: str = Field(min_length=1)

    @field_validator(
        "execution_request_id",
        "tenant_id",
        "selected_operation",
        "qualified_subject_reference",
        "handoff_id",
        "worker_need_id",
        "activated_tool_id",
    )
    @classmethod
    def _validate_non_empty(cls, value: str, info: ValidationInfo) -> str:
        return require_non_empty_text(value, label=str(info.field_name))


class QualifiedToolInvocationMaterialOutcome(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


class QualifiedToolInvocationMaterialResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: QualifiedToolInvocationMaterialOutcome
    material: BaseModel | None = None
    reason_detail: str = ""


@runtime_checkable
class QualifiedToolInvocationMaterialProvider(Protocol):
    """Host/application supplies typed business invocation payload."""

    def provide(
        self,
        request: QualifiedToolInvocationMaterialRequest,
    ) -> QualifiedToolInvocationMaterialResult: ...


@runtime_checkable
class QualifiedToolInvocationResolver(Protocol):
    """Tool-owned mapping from typed material to canonical catalog invoke request."""

    def resolve(
        self,
        *,
        activated_tool_id: str,
        selected_operation: str,
        material: BaseModel,
        tenant_id: str,
        task_id: TaskId,
        run_id: str,
        agent_id: str,
        step_id: str,
        execution_request_id: str,
        correlation_request_id: str | None,
        idempotency_key: str | None,
    ) -> ExecutionBoundCatalogToolInvokeRequest: ...


__all__ = [
    "QualifiedToolInvocationMaterialOutcome",
    "QualifiedToolInvocationMaterialProvider",
    "QualifiedToolInvocationMaterialRequest",
    "QualifiedToolInvocationMaterialResult",
    "QualifiedToolInvocationResolver",
]
