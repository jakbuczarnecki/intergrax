# © Artur Czarnecki. All rights reserved.

"""Delegated execution correlation query service (P2.1-S2C3)."""

from __future__ import annotations

from pydantic import ValidationError

from intergrax.contracts.delegated_execution_query import (
    DELEGATED_EXECUTION_QUERY_INVALID_CURSOR_MESSAGE,
    DELEGATED_EXECUTION_QUERY_VALIDATION_FAILURE_MESSAGE,
    DelegatedExecutionQueryInvalidCursorError,
    DelegatedExecutionQueryPage,
    DelegatedExecutionQueryPort,
    DelegatedExecutionQueryValidationError,
    DelegatedInvocationCorrelationQuery,
    DelegatedInvocationCorrelationQueryStore,
    correlation_view_from_record,
)
from intergrax.contracts.delegated_invocation_correlation import (
    DELEGATED_INVOCATION_CORRELATION_PERSISTENCE_UNAVAILABLE_MESSAGE,
    DelegatedInvocationCorrelationIntegrityError,
    DelegatedInvocationCorrelationPersistenceError,
    DelegatedInvocationCorrelationRecord,
)
from intergrax.runtime.execution.delegated_execution.correlation_query_cursor import (
    encode_delegated_correlation_query_cursor,
)


class DelegatedExecutionQueryService(DelegatedExecutionQueryPort):
    """
    Validates typed queries, loads persisted correlation facts, maps read models.

    Does not call providers or mutate Execution lifecycle or correlation records.
    """

    __slots__ = ("_query_store",)

    def __init__(self, query_store: DelegatedInvocationCorrelationQueryStore) -> None:
        self._query_store = query_store

    def query_delegated_executions(
        self,
        query: DelegatedInvocationCorrelationQuery,
    ) -> DelegatedExecutionQueryPage:
        normalized = _validate_query(query)
        try:
            records = self._query_store.query_correlations(normalized)
        except DelegatedExecutionQueryInvalidCursorError:
            raise
        except DelegatedInvocationCorrelationIntegrityError:
            raise
        except DelegatedInvocationCorrelationPersistenceError:
            raise
        except Exception as exc:
            raise DelegatedInvocationCorrelationPersistenceError(
                DELEGATED_INVOCATION_CORRELATION_PERSISTENCE_UNAVAILABLE_MESSAGE,
            ) from exc

        views = tuple(correlation_view_from_record(record) for record in records)
        if not records:
            return DelegatedExecutionQueryPage(items=(), next_cursor=None, has_more=False)

        last_record = records[-1]
        has_more = self._query_store.has_more_after_page(normalized, last_record)
        next_cursor = None
        if has_more:
            next_cursor = encode_delegated_correlation_query_cursor(
                query=normalized,
                last_persisted_at=last_record.persisted_at,
                last_execution_id=last_record.binding.execution_id,
            )
        return DelegatedExecutionQueryPage(
            items=views,
            next_cursor=next_cursor,
            has_more=has_more,
        )


def _validate_query(
    query: DelegatedInvocationCorrelationQuery,
) -> DelegatedInvocationCorrelationQuery:
    try:
        return DelegatedInvocationCorrelationQuery.model_validate(query.model_dump())
    except ValidationError as exc:
        raise DelegatedExecutionQueryValidationError(
            DELEGATED_EXECUTION_QUERY_VALIDATION_FAILURE_MESSAGE,
        ) from exc


__all__ = ["DelegatedExecutionQueryService"]
