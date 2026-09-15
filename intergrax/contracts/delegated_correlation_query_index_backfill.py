# © Artur Czarnecki. All rights reserved.

"""Bounded resumable backfill of delegated correlation DocumentStore query index fields."""

from __future__ import annotations

from typing import Final, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from intergrax.contracts.delegated_execution_query import (
    DEFAULT_DELEGATED_CORRELATION_QUERY_PAGE_SIZE,
    delegated_correlation_backend_scan_limit,
)

MAX_DELEGATED_CORRELATION_BACKEND_PAGES_PER_BACKFILL_CALL: Final = 1


class DelegatedCorrelationQueryIndexBackfillRequest(BaseModel):
    """One maintenance call: at most ``scan_limit`` backend rows inspected."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scan_limit: int = DEFAULT_DELEGATED_CORRELATION_QUERY_PAGE_SIZE
    cursor: str | None = None

    @field_validator("scan_limit")
    @classmethod
    def _validate_scan_limit(cls, value: int) -> int:
        return delegated_correlation_backend_scan_limit(value)


class DelegatedCorrelationQueryIndexBackfillPage(BaseModel):
    """Result of a single bounded backfill scan window."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scanned_count: int = Field(ge=0)
    updated_count: int = Field(ge=0)
    next_cursor: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_more(self) -> bool:
        return self.next_cursor is not None


@runtime_checkable
class DelegatedCorrelationQueryIndexBackfillPort(Protocol):
    """Pluggable maintenance boundary for correlation query-index backfill."""

    def backfill_correlation_query_index_page(
        self,
        request: DelegatedCorrelationQueryIndexBackfillRequest,
    ) -> DelegatedCorrelationQueryIndexBackfillPage: ...


__all__ = [
    "DelegatedCorrelationQueryIndexBackfillPage",
    "DelegatedCorrelationQueryIndexBackfillPort",
    "DelegatedCorrelationQueryIndexBackfillRequest",
    "MAX_DELEGATED_CORRELATION_BACKEND_PAGES_PER_BACKFILL_CALL",
]
