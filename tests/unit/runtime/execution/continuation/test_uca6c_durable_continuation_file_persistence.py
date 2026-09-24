# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R5.9-R1-R1 — durable continuation file persistence across process boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
    ExecutionContinuationIdentity,
    ExecutionPauseRequest,
)
from intergrax.contracts.governed_continuation_correlation import ContinuationReason
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id
from intergrax.runtime.execution.continuation.composition import (
    reconnect_execution_engine_continuation_dependencies,
    wire_execution_engine_continuation_dependencies,
)
from intergrax.runtime.execution.continuation.durable_state_file import (
    ExecutionContinuationDurableStateFilePersistence,
)
from intergrax.runtime.execution.continuation.persistence import (
    ExecutionContinuationDurableBacking,
    backing_execution_continuation_state_store,
)
from testing_support.builder import (
    canonical_run_id_for_tests,
    canonical_task_id_for_tests,
)

pytestmark = pytest.mark.unit

_TASK = canonical_task_id_for_tests("uca6c-cont-file")
_RUN = canonical_run_id_for_tests("uca6c-cont-file")
_ATTEMPT = mint_attempt_id()
_EXECUTION = mint_execution_id()
_CONTINUATION_ID = "cont_uca6c_file_1"


def _identity() -> ExecutionContinuationIdentity:
    return ExecutionContinuationIdentity(
        task_id=_TASK,
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXECUTION,
    )


def _pause_on_host_a(backing: ExecutionContinuationDurableBacking) -> None:
    store = backing_execution_continuation_state_store(backing)
    deps = wire_execution_engine_continuation_dependencies(state_store=store)
    deps.continuation_service.request_pause(
        ExecutionPauseRequest(
            identity=_identity(),
            continuation_id=_CONTINUATION_ID,
            reason=ContinuationReason.SECURITY,
            pause_id="pause_uca6c_file",
            human_request_id="hr_uca6c_file",
        ),
    )


def test_host_a_file_persist_host_b_load_same_identity(tmp_path: Path) -> None:
    path = tmp_path / "continuation.json"
    persistence = ExecutionContinuationDurableStateFilePersistence(path)
    backing_a = ExecutionContinuationDurableBacking()
    _pause_on_host_a(backing_a)
    persistence.persist_from_backing(backing_a)
    del backing_a

    store_b = persistence.load_state_store()
    deps_b = reconnect_execution_engine_continuation_dependencies(state_store=store_b)
    located = deps_b.continuation_service.store.load(_CONTINUATION_ID)
    assert located is not None
    assert located.identity == _identity()
    current = deps_b.continuation_service.store.resolve_current_episode_for_identity(
        _identity(),
    )
    assert current is not None
    assert current.continuation_id == _CONTINUATION_ID


def test_missing_continuation_file_fail_closed(tmp_path: Path) -> None:
    persistence = ExecutionContinuationDurableStateFilePersistence(
        tmp_path / "missing.json",
    )
    with pytest.raises(ExecutionContinuationError) as exc:
        persistence.load_state_store()
    assert exc.value.code is ExecutionContinuationErrorCode.NOT_FOUND


def test_malformed_continuation_file_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not-json", encoding="utf-8")
    persistence = ExecutionContinuationDurableStateFilePersistence(path)
    with pytest.raises(ExecutionContinuationError) as exc:
        persistence.load_state_store()
    assert exc.value.code is ExecutionContinuationErrorCode.CORRUPT_CONTINUATION_STATE
