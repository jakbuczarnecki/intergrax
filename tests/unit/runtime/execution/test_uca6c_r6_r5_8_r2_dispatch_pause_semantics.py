# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.8-R2 — dispatch maps governed pause to DISPATCHED correlation."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionDispatchDisposition,
)
from intergrax.contracts.execution_intake import CanonicalExecutionInvocationFailed
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id, mint_run_id
from intergrax.runtime.execution.canonical_dispatch_invocation_outcome import (
    qualified_dispatch_result_for_invocation_failure,
)
from intergrax.runtime.execution.suspended_operation.pause_required import (
    ExecutionSuspendedWorkPauseRequired,
)

pytestmark = pytest.mark.unit


def test_pause_invocation_failure_maps_to_dispatched() -> None:
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    pause = ExecutionSuspendedWorkPauseRequired.__new__(
        ExecutionSuspendedWorkPauseRequired,
    )
    exc = CanonicalExecutionInvocationFailed(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        cause=pause,
    )
    result = qualified_dispatch_result_for_invocation_failure(
        exc,
        execution_request_id="worker-qualified-capability-execution:test",
    )
    assert result.disposition is QualifiedCapabilityExecutionDispatchDisposition.DISPATCHED
    assert result.execution_id == execution_id
    assert result.run_id == run_id
    assert result.attempt_id == attempt_id


def test_non_pause_invocation_failure_still_raises() -> None:
    exc = CanonicalExecutionInvocationFailed(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        cause=RuntimeError("backend_fail"),
    )
    with pytest.raises(CanonicalExecutionInvocationFailed):
        qualified_dispatch_result_for_invocation_failure(
            exc,
            execution_request_id="x",
        )
