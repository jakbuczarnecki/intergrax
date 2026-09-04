# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Durable attempt lifecycle authority (P0C-4)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.execution_identity import AttemptId, RunId


class AttemptTransitionReason(StrEnum):
    INITIAL = "initial"
    RETRY = "retry"


class AttemptLifecycleState(BaseModel):
    """Canonical durable attempt lifecycle record for one Run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: RunId
    active_attempt_id: AttemptId
    previous_attempt_id: AttemptId | None = None
    generation: int = Field(ge=1)
    transition_reason: AttemptTransitionReason | None = None


class AttemptTransitionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: RunId
    previous_attempt_id: AttemptId
    active_attempt_id: AttemptId
    generation: int = Field(ge=1)


class AttemptLifecycleError(RuntimeError):
    """Raised when durable attempt lifecycle state is missing or corrupt."""


class AttemptLifecycleStore(ABC):
    """Provider-neutral durable attempt lifecycle persistence."""

    @property
    @abstractmethod
    def is_durable(self) -> bool:
        """Whether state survives process restart."""

    @abstractmethod
    def load_raw(self, *, tenant_id: str, run_id: RunId) -> bytes | None:
        """Return encoded lifecycle bytes or ``None`` when absent."""

    @abstractmethod
    def compare_and_swap(
        self,
        *,
        tenant_id: str,
        run_id: RunId,
        expected: bytes | None,
        new_state: AttemptLifecycleState,
    ) -> bool:
        """Atomically replace lifecycle bytes when ``expected`` matches."""
