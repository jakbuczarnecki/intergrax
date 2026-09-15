# © Artur Czarnecki. All rights reserved.

"""Delegated invocation correlation service (P2.1-S2C)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from intergrax.contracts.delegated_execution_invocation_binding import (
    DelegatedExecutionInvocationBinding,
)
from intergrax.contracts.delegated_invocation_correlation import (
    DELEGATED_INVOCATION_CORRELATION_NOT_FOUND_MESSAGE,
    DelegatedInvocationCorrelationConflictError,
    DelegatedInvocationCorrelationIntegrityError,
    DelegatedInvocationCorrelationNotFoundError,
    DelegatedInvocationCorrelationPersistenceError,
    DelegatedInvocationCorrelationRecord,
    DelegatedInvocationCorrelationStore,
)
from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id


@runtime_checkable
class DelegatedInvocationCorrelationLookup(Protocol):
    """Read-only durable binding lookup for control and recovery paths."""

    def load_binding_by_execution_id(
        self,
        execution_id: ExecutionId,
    ) -> DelegatedExecutionInvocationBinding:
        ...


class DelegatedInvocationCorrelationService(DelegatedInvocationCorrelationLookup):
    """Persists and loads platform-issued invocation bindings."""

    __slots__ = ("_store",)

    def __init__(self, store: DelegatedInvocationCorrelationStore) -> None:
        self._store = store

    @property
    def store(self) -> DelegatedInvocationCorrelationStore:
        return self._store

    def persist_binding(
        self,
        binding: DelegatedExecutionInvocationBinding,
        *,
        persisted_at: datetime | None = None,
    ) -> None:
        when = persisted_at or datetime.now(timezone.utc)
        if when.tzinfo is None:
            raise ValueError("persisted_at must be timezone-aware")
        record = DelegatedInvocationCorrelationRecord(
            binding=binding,
            persisted_at=when,
        )
        try:
            self._store.persist(record)
        except (
            DelegatedInvocationCorrelationIntegrityError,
            DelegatedInvocationCorrelationConflictError,
            DelegatedInvocationCorrelationPersistenceError,
        ):
            raise
        except Exception as exc:
            raise DelegatedInvocationCorrelationPersistenceError(
                "delegated invocation correlation persistence failed",
            ) from exc

    def load_binding_by_execution_id(
        self,
        execution_id: ExecutionId,
    ) -> DelegatedExecutionInvocationBinding:
        normalized = validate_execution_id(execution_id)
        try:
            record = self._store.get_by_execution_id(normalized)
        except DelegatedInvocationCorrelationIntegrityError:
            raise
        except Exception as exc:
            raise DelegatedInvocationCorrelationPersistenceError(
                "delegated invocation correlation load failed",
            ) from exc
        if record is None:
            raise DelegatedInvocationCorrelationNotFoundError(
                DELEGATED_INVOCATION_CORRELATION_NOT_FOUND_MESSAGE,
            )
        return record.binding


__all__ = [
    "DelegatedInvocationCorrelationLookup",
    "DelegatedInvocationCorrelationService",
]
