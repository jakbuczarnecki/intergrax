# © Artur Czarnecki. All rights reserved.

"""Production composition requirements for canonical execution continuation (GR-10-R12-R1)."""

from __future__ import annotations

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
)
from intergrax.contracts.execution_continuation_state_store import ExecutionContinuationStateStore

MISSING_CONTINUATION_STORE_MSG = (
    "production composition requires an explicit ExecutionContinuationStateStore; "
    "silent in-memory continuation downgrade is lab-only"
)

NON_DURABLE_CONTINUATION_STORE_MSG = (
    "production composition requires a durable ExecutionContinuationStateStore "
    "(is_durable must be True); explicit non-durable stores are lab-only"
)

# Backward-compatible alias used by existing R12 static gates / docs.
DURABLE_CONTINUATION_EXPLICIT_STORE_REQUIRED_MSG = MISSING_CONTINUATION_STORE_MSG


def validate_execution_continuation_for_composition(
    *,
    production_mode: bool,
    state_store: ExecutionContinuationStateStore | None,
    continuation_explicitly_wired: bool,
    continuation_disabled: bool,
) -> None:
    """Fail closed when production continuation lacks a restart-safe store.

    Lab / test hosts (``production_mode=False``) may omit the store and receive the
    documented in-memory default. Production must inject a durable store
    (``store.is_durable is True``) or disable continuation intentionally.

    ``continuation_explicitly_wired`` does **not** bypass durability: an explicit
    port+lifecycle pair still requires a publicly resolvable durable store
    (injected store or ``ExecutionContinuationService.store``).
    """
    if not production_mode or continuation_disabled:
        return
    _ = continuation_explicitly_wired  # API retained; never skips durability.
    if state_store is None:
        raise ExecutionContinuationError(
            MISSING_CONTINUATION_STORE_MSG,
            code=ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE,
        )
    if not state_store.is_durable:
        raise ExecutionContinuationError(
            NON_DURABLE_CONTINUATION_STORE_MSG,
            code=ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE,
        )


__all__ = [
    "DURABLE_CONTINUATION_EXPLICIT_STORE_REQUIRED_MSG",
    "MISSING_CONTINUATION_STORE_MSG",
    "NON_DURABLE_CONTINUATION_STORE_MSG",
    "validate_execution_continuation_for_composition",
]
