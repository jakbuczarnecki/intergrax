# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Durable execution terminal authority (P0C-5)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.execution_identity import RunId


class ExecutionTerminalOutcome(StrEnum):
    """Terminal execution outcomes that block future resume."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionTerminalPersistenceProvider(StrEnum):
    """Composition-time selector for durable :class:`ExecutionTerminalStore` wiring."""

    KV = "kv"
    DOCUMENT_STORE = "document_store"
    CHECKPOINT = "checkpoint"


class ExecutionTerminalRecord(BaseModel):
    """Immutable durable fact that a task execution reached a terminal outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    task_id: str
    run_id: RunId | None = None
    outcome: ExecutionTerminalOutcome
    reason: str = Field(default="", max_length=512)
    recorded_at_utc: str


class ExecutionTerminalError(RuntimeError):
    """Raised when durable terminal authority is missing, corrupt, or unavailable."""


class AmbiguousExecutionTerminalProviderError(ExecutionTerminalError):
    """Raised when multiple durable terminal providers are available without explicit selection."""


class ExecutionTerminalConflictError(ExecutionTerminalError):
    """Raised when a late terminal transition conflicts with an existing terminal record."""

    __slots__ = ("existing_outcome",)

    def __init__(
        self,
        message: str,
        *,
        existing_outcome: ExecutionTerminalOutcome | None = None,
    ) -> None:
        super().__init__(message)
        self.existing_outcome = existing_outcome


@runtime_checkable
class ExecutionTerminalPersistenceCapability(Protocol):
    """Checkpoint-side persistence that can store terminal cancellation authority."""

    def get_terminal_record(self, *, tenant_id: str, task_id: str) -> ExecutionTerminalRecord | None:
        ...

    def put_terminal_record_if_absent(self, record: ExecutionTerminalRecord) -> bool:
        ...


class ExecutionTerminalStore(ABC):
    """Provider-neutral durable terminal execution persistence."""

    @property
    @abstractmethod
    def is_durable(self) -> bool:
        """Whether terminal state survives process restart."""

    @abstractmethod
    def load_record(self, *, tenant_id: str, task_id: str) -> ExecutionTerminalRecord | None:
        """Return the terminal record or ``None`` when absent."""

    @abstractmethod
    def put_if_absent(self, record: ExecutionTerminalRecord) -> bool:
        """Persist ``record`` only when no terminal row exists yet."""
