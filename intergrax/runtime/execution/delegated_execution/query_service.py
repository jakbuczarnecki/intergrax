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
)
from intergrax.runtime.execution.delegated_execution.correlation_query_cursor import (
    DelegatedCorrelationQueryCursorCodec,
    encode_delegated_correlation_query_cursor,
)


class DelegatedExecutionQueryService(DelegatedExecutionQueryPort):
    """
    Validates typed queries, loads persisted correlation facts, maps read models.

    Does not call providers or mutate Execution lifecycle or correlation records.
    """

    __slots__ = ("_cursor_codec", "_query_store")

    def __init__(
        self,
        query_store: DelegatedInvocationCorrelationQueryStore,
        *,
        cursor_codec: DelegatedCorrelationQueryCursorCodec | None = None,
    ) -> None:
        self._query_store = query_store
        self._cursor_codec = cursor_codec or query_store.cursor_codec

    def query_delegated_executions(
        self,
        query: DelegatedInvocationCorrelationQuery,
    ) -> DelegatedExecutionQueryPage:
        normalized = _validate_query(query)
        try:
            store_page = self._query_store.query_page(normalized)
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

        records = store_page.records
        views = tuple(correlation_view_from_record(record) for record in records)
        if not store_page.has_more:
            return DelegatedExecutionQueryPage(
                items=views,
                next_cursor=None,
                has_more=False,
            )

        last_record = records[-1] if records else None
        next_cursor = encode_delegated_correlation_query_cursor(
            codec=self._cursor_codec,
            query=normalized,
            last_persisted_at=last_record.persisted_at if last_record is not None else None,
            last_execution_id=(
                last_record.binding.execution_id if last_record is not None else None
            ),
            document_store_cursor=store_page.backend_continuation_cursor,
        )
        return DelegatedExecutionQueryPage(
            items=views,
            next_cursor=next_cursor,
            has_more=True,
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
