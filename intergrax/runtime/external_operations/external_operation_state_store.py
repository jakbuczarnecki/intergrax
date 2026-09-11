# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Durable external operation state with revision CAS (W4-C)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationIntentState,
    ExternalOperationNotFoundError,
    ExternalOperationPhysicalState,
    ExternalOperationState,
)


class StaleExternalOperationStateError(RuntimeError):
    """CAS conflict on external operation durable state."""

    def __init__(
        self,
        *,
        operation_id: str,
        expected_revision: int,
        actual_revision: int | None,
    ) -> None:
        self.operation_id = operation_id
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision
        super().__init__(
            f"stale external operation state for {operation_id!r}: "
            f"expected revision {expected_revision}, actual {actual_revision!r}",
        )


@dataclass(frozen=True, slots=True)
class ExternalOperationStateUpdate:
    intent_state: ExternalOperationIntentState | None = None
    physical_state: ExternalOperationPhysicalState | None = None
    owner_token: str | None = None
    clear_owner_token: bool = False


@runtime_checkable
class ExternalOperationStateStore(Protocol):
    """Process-local or shared durable store for external operation lifecycle."""

    def load(self, operation_id: str) -> ExternalOperationState | None:
        ...

    def create_if_absent(
        self,
        operation_id: str,
        *,
        owner_token: str | None = None,
    ) -> ExternalOperationState:
        ...

    def compare_and_set(
        self,
        operation_id: str,
        *,
        expected_revision: int,
        update: ExternalOperationStateUpdate,
    ) -> ExternalOperationState:
        ...

    def list_running(self) -> tuple[ExternalOperationState, ...]:
        ...


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class InMemoryExternalOperationStateStore:
    """Thread-safe in-memory store for tests and process-local reconciliation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, ExternalOperationState] = {}

    def load(self, operation_id: str) -> ExternalOperationState | None:
        if type(operation_id) is not str or not operation_id:
            raise ValueError("operation_id must be a non-empty str")
        with self._lock:
            return self._records.get(operation_id)

    def create_if_absent(
        self,
        operation_id: str,
        *,
        owner_token: str | None = None,
    ) -> ExternalOperationState:
        if type(operation_id) is not str or not operation_id:
            raise ValueError("operation_id must be a non-empty str")
        now = _utc_now()
        with self._lock:
            existing = self._records.get(operation_id)
            if existing is not None:
                return existing
            record = ExternalOperationState(
                operation_id=operation_id,
                intent_state=ExternalOperationIntentState.ACTIVE,
                physical_state=ExternalOperationPhysicalState.NOT_STARTED,
                created_at=now,
                updated_at=now,
                revision=1,
                owner_token=owner_token,
            )
            self._records[operation_id] = record
            return record

    def compare_and_set(
        self,
        operation_id: str,
        *,
        expected_revision: int,
        update: ExternalOperationStateUpdate,
    ) -> ExternalOperationState:
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("expected_revision must be int >= 1")
        now = _utc_now()
        with self._lock:
            current = self._records.get(operation_id)
            if current is None:
                raise ExternalOperationNotFoundError(operation_id)
            if current.revision != expected_revision:
                raise StaleExternalOperationStateError(
                    operation_id=operation_id,
                    expected_revision=expected_revision,
                    actual_revision=current.revision,
                )
            intent = (
                update.intent_state
                if update.intent_state is not None
                else current.intent_state
            )
            physical = (
                update.physical_state
                if update.physical_state is not None
                else current.physical_state
            )
            owner = current.owner_token
            if update.clear_owner_token:
                owner = None
            elif update.owner_token is not None:
                owner = update.owner_token
            next_record = ExternalOperationState(
                operation_id=operation_id,
                intent_state=intent,
                physical_state=physical,
                created_at=current.created_at,
                updated_at=now,
                revision=current.revision + 1,
                owner_token=owner,
            )
            self._records[operation_id] = next_record
            return next_record

    def list_running(self) -> tuple[ExternalOperationState, ...]:
        with self._lock:
            return tuple(
                record
                for record in self._records.values()
                if record.physical_state is ExternalOperationPhysicalState.RUNNING
            )
