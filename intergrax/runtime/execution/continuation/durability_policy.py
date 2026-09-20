# © Artur Czarnecki. All rights reserved.

"""Production composition requirements for canonical execution continuation (GR-10-R12)."""

from __future__ import annotations

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
)
from intergrax.contracts.execution_continuation_state_store import ExecutionContinuationStateStore

DURABLE_CONTINUATION_EXPLICIT_STORE_REQUIRED_MSG = (
    "production composition requires an explicit ExecutionContinuationStateStore; "
    "silent in-memory continuation downgrade is lab-only"
)


def validate_execution_continuation_for_composition(
    *,
    production_mode: bool,
    state_store: ExecutionContinuationStateStore | None,
    continuation_explicitly_wired: bool,
    continuation_disabled: bool,
) -> None:
    """Fail closed when production silently defaults continuation to an implicit lab store.

    Lab / test hosts (``production_mode=False``) may omit the store and receive the
    documented in-memory default. Production must inject an explicit store or an
    explicit port+lifecycle pair, or disable continuation intentionally.
    """
    if not production_mode or continuation_disabled:
        return
    if continuation_explicitly_wired:
        return
    if state_store is None:
        raise ExecutionContinuationError(
            DURABLE_CONTINUATION_EXPLICIT_STORE_REQUIRED_MSG,
            code=ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE,
        )


__all__ = [
    "DURABLE_CONTINUATION_EXPLICIT_STORE_REQUIRED_MSG",
    "validate_execution_continuation_for_composition",
]
