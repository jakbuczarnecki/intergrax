# © Artur Czarnecki. All rights reserved.

"""Delegated execution correlation list/query read model (P2.1-S2C3).

Queries return **known persisted delegated correlations** only. They do not
describe canonical Execution lifecycle state, provider live status, or
continuability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.delegated_invocation_correlation import (
    DelegatedInvocationCorrelationRecord,
)
from intergrax.contracts.execution_identity import (
    ExecutionId,
    RunId,
    validate_execution_id,
    validate_run_id,
)

SCHEMA_DELEGATED_INVOCATION_CORRELATION_QUERY_V1: Final = (
    "delegated_invocation_correlation_query.v1"
)
SCHEMA_DELEGATED_EXECUTION_CORRELATION_VIEW_V1: Final = (
    "delegated_execution_correlation_view.v1"
)

DEFAULT_DELEGATED_CORRELATION_QUERY_PAGE_SIZE: Final = 100
MAX_DELEGATED_CORRELATION_QUERY_PAGE_SIZE: Final = 500

DELEGATED_EXECUTION_QUERY_INVALID_CURSOR_MESSAGE: Final = (
    "delegated execution query cursor is invalid"
)
DELEGATED_EXECUTION_QUERY_VALIDATION_FAILURE_MESSAGE: Final = (
    "delegated execution query failed validation"
)

_NON_EMPTY = Field(min_length=1)


class DelegatedExecutionQueryError(RuntimeError):
    """Base error for delegated execution correlation queries."""


class DelegatedExecutionQueryValidationError(DelegatedExecutionQueryError):
    """Raised when query parameters fail platform validation."""


class DelegatedExecutionQueryInvalidCursorError(DelegatedExecutionQueryError):
    """Raised when a pagination cursor is invalid or scope-mismatched."""


class DelegatedInvocationCorrelationQuery(BaseModel):
    """Bounded, typed filters over persisted correlation evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["delegated_invocation_correlation_query.v1"] = (
        SCHEMA_DELEGATED_INVOCATION_CORRELATION_QUERY_V1
    )
    page_size: int = DEFAULT_DELEGATED_CORRELATION_QUERY_PAGE_SIZE
    cursor: str | None = None
    parent_execution_id: ExecutionId | None = None
    provider_id: str | None = None
    run_id: RunId | None = None
    persisted_from: datetime | None = None
    persisted_to: datetime | None = None

    @field_validator("page_size")
    @classmethod
    def _validate_page_size(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("page_size must be a positive integer")
        if value < 1 or value > MAX_DELEGATED_CORRELATION_QUERY_PAGE_SIZE:
            raise ValueError("page_size out of bounds")
        return value

    @field_validator("parent_execution_id", mode="before")
    @classmethod
    def _validate_parent(cls, value: object | None) -> ExecutionId | None:
        if value is None:
            return None
        return validate_execution_id(value)

    @field_validator("run_id", mode="before")
    @classmethod
    def _validate_run(cls, value: object | None) -> RunId | None:
        if value is None:
            return None
        return validate_run_id(value)

    @field_validator("provider_id")
    @classmethod
    def _validate_provider_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("provider_id must be non-empty when set")
        return normalized

    @field_validator("persisted_from", "persisted_to")
    @classmethod
    def _require_tz_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("persisted time bounds must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_time_window(self) -> DelegatedInvocationCorrelationQuery:
        if (
            self.persisted_from is not None
            and self.persisted_to is not None
            and self.persisted_from > self.persisted_to
        ):
            raise ValueError("persisted_from must not be after persisted_to")
        return self


class DelegatedExecutionCorrelationView(BaseModel):
    """Read-only correlation facts derived from a persisted correlation record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["delegated_execution_correlation_view.v1"] = (
        SCHEMA_DELEGATED_EXECUTION_CORRELATION_VIEW_V1
    )
    execution_id: ExecutionId
    parent_execution_id: ExecutionId
    run_id: RunId
    provider_id: str = _NON_EMPTY
    invocation_id: str = _NON_EMPTY
    provider_request_id: str = _NON_EMPTY
    provider_operation_id: str = _NON_EMPTY
    persisted_at: datetime

    @field_validator("persisted_at")
    @classmethod
    def _require_tz_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("persisted_at must be timezone-aware")
        return value


class DelegatedExecutionQueryPage(BaseModel):
    """One bounded page of persisted delegated correlation facts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[DelegatedExecutionCorrelationView, ...]
    next_cursor: str | None = None
    has_more: bool = False


class DelegatedInvocationCorrelationQueryStore(ABC):
    """Read-only, provider-neutral query port over persisted correlation records."""

    @abstractmethod
    def query_correlations(
        self,
        query: DelegatedInvocationCorrelationQuery,
    ) -> tuple[DelegatedInvocationCorrelationRecord, ...]:
        """
        Return up to ``query.page_size`` records for this page.

        Implementations must apply filters with AND semantics, order by
        ``persisted_at`` descending with ``execution_id`` tie-breaker, and
        honor ``query.cursor`` when continuing a page.
        """

    @abstractmethod
    def has_more_after_page(
        self,
        query: DelegatedInvocationCorrelationQuery,
        last_record: DelegatedInvocationCorrelationRecord,
    ) -> bool:
        """Return whether another page exists after ``last_record`` for ``query``."""


@runtime_checkable
class DelegatedExecutionQueryPort(Protocol):
    """Consumer-facing delegated correlation list/query port."""

    def query_delegated_executions(
        self,
        query: DelegatedInvocationCorrelationQuery,
    ) -> DelegatedExecutionQueryPage:
        ...


def correlation_view_from_record(
    record: DelegatedInvocationCorrelationRecord,
) -> DelegatedExecutionCorrelationView:
    binding = record.binding
    inv = binding.provider_invocation
    return DelegatedExecutionCorrelationView(
        execution_id=binding.execution_id,
        parent_execution_id=binding.parent_execution_id,
        run_id=binding.run_id,
        provider_id=binding.provider_id,
        invocation_id=inv.invocation_id,
        provider_request_id=inv.provider_request_id,
        provider_operation_id=inv.provider_operation_id,
        persisted_at=record.persisted_at,
    )


__all__ = [
    "DEFAULT_DELEGATED_CORRELATION_QUERY_PAGE_SIZE",
    "DELEGATED_EXECUTION_QUERY_INVALID_CURSOR_MESSAGE",
    "DELEGATED_EXECUTION_QUERY_VALIDATION_FAILURE_MESSAGE",
    "DelegatedExecutionCorrelationView",
    "DelegatedExecutionQueryError",
    "DelegatedExecutionQueryInvalidCursorError",
    "DelegatedExecutionQueryPage",
    "DelegatedExecutionQueryPort",
    "DelegatedExecutionQueryValidationError",
    "DelegatedInvocationCorrelationQuery",
    "DelegatedInvocationCorrelationQueryStore",
    "MAX_DELEGATED_CORRELATION_QUERY_PAGE_SIZE",
    "SCHEMA_DELEGATED_EXECUTION_CORRELATION_VIEW_V1",
    "SCHEMA_DELEGATED_INVOCATION_CORRELATION_QUERY_V1",
    "correlation_view_from_record",
]
