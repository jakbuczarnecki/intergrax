# © Artur Czarnecki. All rights reserved.

"""Production composition-owned durable execution continuation store factory (PROD-Q-R1-DUR-02)."""

from __future__ import annotations

from pathlib import Path

from intergrax.runtime.execution.continuation.durable_serialized_seal_store import (
    MissingContinuationFilePolicy,
    SerializedSealExecutionContinuationStateStore,
    open_serialized_seal_execution_continuation_state_store,
)


def open_production_execution_continuation_state_store(
    state_file: Path,
    *,
    missing_file_policy: MissingContinuationFilePolicy = "fail",
) -> SerializedSealExecutionContinuationStateStore:
    """
    Open the sanctioned serialized-seal continuation store for one production process.

    The caller (composition root / deploy-activate launcher) must supply an explicit
    ``state_file`` path, invoke exactly once per process, and call ``close()`` at
    authorized process shutdown. Single-writer: at most one live process per file.
    Atomic rename only — no power-loss durability claim without separate sync evidence.
    """
    return open_serialized_seal_execution_continuation_state_store(
        state_file,
        missing_file_policy=missing_file_policy,
    )


__all__ = [
    "MissingContinuationFilePolicy",
    "SerializedSealExecutionContinuationStateStore",
    "open_production_execution_continuation_state_store",
]
