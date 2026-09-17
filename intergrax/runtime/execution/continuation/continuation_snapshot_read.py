# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Read-only continuation snapshot access for inspection consumers."""

from __future__ import annotations

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationIdentity,
    PendingExecutionContinuation,
)
from intergrax.contracts.execution_continuation_read import ExecutionContinuationSnapshotReadPort
from intergrax.contracts.execution_continuation_state_store import ExecutionContinuationStateStore


class ExecutionContinuationStateStoreReadAdapter:
    """ExecutionContinuationSnapshotReadPort over canonical state store reads."""

    def __init__(
        self,
        store: ExecutionContinuationStateStore,
        *,
        source_id: str = "execution_continuation_state_store",
    ) -> None:
        self._store = store
        self._source_id = source_id

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def is_durable(self) -> bool:
        return self._store.is_durable

    def read_current_episode(
        self,
        identity: ExecutionContinuationIdentity,
    ) -> PendingExecutionContinuation | None:
        return self._store.resolve_current_episode_for_identity(identity)


__all__ = ["ExecutionContinuationStateStoreReadAdapter"]
