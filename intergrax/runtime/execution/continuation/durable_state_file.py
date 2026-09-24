# © Artur Czarnecki. All rights reserved.

"""File-backed durable persistence for serialized execution continuation state."""

from __future__ import annotations

import json
from pathlib import Path

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
)
from intergrax.contracts.structured_json_value import StructuredJsonObject
from intergrax.runtime.execution.continuation.persistence import (
    ExecutionContinuationDurableBacking,
    ReconstructedDurableExecutionContinuationStateStore,
    execution_continuation_state_store_from_durable_export,
    export_durable_continuation_state,
    restore_durable_continuation_backing,
)


class ExecutionContinuationDurableStateFilePersistence:
    """Pluginable file persistence for official durable continuation export envelope."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def persist_from_backing(
        self,
        backing: ExecutionContinuationDurableBacking,
    ) -> None:
        payload = export_durable_continuation_state(backing)
        self._atomic_write(payload)

    def load_export_payload(self) -> StructuredJsonObject:
        if not self._path.is_file():
            raise ExecutionContinuationError(
                "durable continuation state file is missing",
                code=ExecutionContinuationErrorCode.NOT_FOUND,
            )
        try:
            raw = self._path.read_text(encoding="utf-8")
            payload = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise ExecutionContinuationError(
                "durable continuation state file is unreadable or malformed",
                code=ExecutionContinuationErrorCode.CORRUPT_CONTINUATION_STATE,
            ) from exc
        if not isinstance(payload, dict):
            raise ExecutionContinuationError(
                "durable continuation state envelope must be a JSON object",
                code=ExecutionContinuationErrorCode.CORRUPT_CONTINUATION_STATE,
            )
        return payload

    def load_state_store(
        self,
    ) -> ReconstructedDurableExecutionContinuationStateStore:
        return execution_continuation_state_store_from_durable_export(
            self.load_export_payload(),
        )

    def restore_backing(self) -> ExecutionContinuationDurableBacking:
        return restore_durable_continuation_backing(self.load_export_payload())

    def _atomic_write(self, payload: StructuredJsonObject) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, indent=2, sort_keys=True)
        temp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        try:
            temp_path.write_text(serialized, encoding="utf-8")
            temp_path.replace(self._path)
        except OSError as exc:
            raise ExecutionContinuationError(
                "failed to persist durable continuation state",
                code=ExecutionContinuationErrorCode.STORE_QUERY_FAILED,
            ) from exc


__all__ = ["ExecutionContinuationDurableStateFilePersistence"]
