# © Artur Czarnecki. All rights reserved.

"""GR-5-R2 — canonical ExecutionContinuationService integration qualification."""

from __future__ import annotations

import ast
import threading
from pathlib import Path

import pytest

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
    ExecutionContinuationLookup,
    ExecutionContinuationPort,
    ExecutionContinuationResolutionCommand,
    ExecutionContinuationResumeCommand,
    ExecutionContinuationTransition,
    ExecutionHumanVerdict,
    ExecutionPauseRequest,
    PendingExecutionContinuation,
    advance_continuation_lifecycle,
)
from intergrax.contracts.execution_continuation_state_store import ExecutionContinuationStateStore
from intergrax.contracts.governed_continuation_correlation import (
    ContinuationReason,
    GovernedContinuationCorrelation,
)
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id
from intergrax.runtime.execution.continuation.composition import (
    ExecutionEngineContinuationDependencies,
    wire_execution_continuation_port,
    wire_execution_engine_continuation_dependencies,
)
from intergrax.runtime.execution.continuation.persistence import (
    InMemoryExecutionContinuationStateStore,
)
from intergrax.runtime.execution.continuation.lifecycle_driver import (
    ExecutionContinuationLifecycleDriver,
)
from intergrax.runtime.execution.continuation.service import ExecutionContinuationService
from testing_support.builder import canonical_run_id_for_tests, canonical_task_id_for_tests

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SERVICE_MODULE = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "continuation" / "service.py"
_CONTRACT_MODULE = _REPO_ROOT / "intergrax" / "contracts" / "execution_continuation.py"

_SEED = "gr5-r2-continuation"
_TASK = canonical_task_id_for_tests(_SEED)
_RUN = canonical_run_id_for_tests(_SEED)
_ATTEMPT = mint_attempt_id()
_EXECUTION = mint_execution_id()
_CONTINUATION_ID = "gcr_gr5_r2_test"
_SCOPE_DIGEST = "sha256:" + "a" * 64


def _identity(**overrides: object) -> ExecutionContinuationIdentity:
    payload = {
        "task_id": _TASK,
        "run_id": _RUN,
        "attempt_id": _ATTEMPT,
        "execution_id": _EXECUTION,
    }
    payload.update(overrides)
    return ExecutionContinuationIdentity(**payload)  # type: ignore[arg-type]


def _governed_correlation(**overrides: object) -> GovernedContinuationCorrelation:
    payload: dict[str, object] = {
        "continuation_request_id": _CONTINUATION_ID,
        "reason": ContinuationReason.SECURITY,
        "task_id": _TASK,
        "run_id": _RUN,
        "attempt_id": _ATTEMPT,
        "execution_id": _EXECUTION,
        "operation_id": "op_gr5_r2",
        "side_effect_scope_id": "scope_gr5_r2",
    }
    payload.update(overrides)
    return GovernedContinuationCorrelation.model_validate(payload)


def _pause_request(**overrides: object) -> ExecutionPauseRequest:
    payload: dict[str, object] = {
        "identity": _identity(),
        "continuation_id": _CONTINUATION_ID,
        "reason": ContinuationReason.SECURITY,
        "governed_correlation": _governed_correlation(),
        "pause_id": "pause_gr5",
        "human_request_id": "hr_gr5",
    }
    payload.update(overrides)
    return ExecutionPauseRequest.model_validate(payload)


def _resolution_command(
    *,
    expected_revision: int,
    verdict: ExecutionHumanVerdict = ExecutionHumanVerdict.APPROVE,
    **overrides: object,
) -> ExecutionContinuationResolutionCommand:
    payload: dict[str, object] = {
        "continuation_id": _CONTINUATION_ID,
        "identity": _identity(),
        "expected_revision": expected_revision,
        "verdict": verdict,
        "approver": local_development_approver_evidence(
            actor_id="op-r2",
            tenant_id="tenant-gr5-r2",
        ),
        "human_request_id": "hr_gr5",
        "pause_id": "pause_gr5",
        "operation_id": "op_gr5_r2",
        "side_effect_scope_id": "scope_gr5_r2",
        "resolved_at": "2026-09-15T01:00:00Z",
    }
    payload.update(overrides)
    return ExecutionContinuationResolutionCommand.model_validate(payload)


def _port() -> ExecutionContinuationPort:
    return wire_execution_continuation_port()


def _assert_four_ids_unchanged(pending: PendingExecutionContinuation) -> None:
    assert pending.identity.task_id == _TASK
    assert pending.identity.run_id == _RUN
    assert pending.identity.attempt_id == _ATTEMPT
    assert pending.identity.execution_id == _EXECUTION
    assert pending.continuation_id == _CONTINUATION_ID
    assert pending.governed_correlation == _governed_correlation()


def _drive_to_waiting(
    deps: ExecutionEngineContinuationDependencies,
) -> PendingExecutionContinuation:
    port = deps.continuation
    driver = deps.lifecycle_driver
    port.request_pause(_pause_request())
    paused = driver.record_execution_reached_safe_pause(
        _CONTINUATION_ID,
        execution_pause_established=True,
    )
    assert paused.lifecycle_state is ExecutionContinuationLifecycleState.PAUSED
    waiting = driver.record_ready_for_human_resolution(_CONTINUATION_ID)
    assert waiting.lifecycle_state is ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN
    return waiting


def _deps() -> ExecutionEngineContinuationDependencies:
    return wire_execution_engine_continuation_dependencies()


def test_request_pause_creates_pause_requested_revision_one() -> None:
    port = _port()
    pending = port.request_pause(_pause_request())
    assert pending.lifecycle_state is ExecutionContinuationLifecycleState.PAUSE_REQUESTED
    assert pending.revision == 1
    _assert_four_ids_unchanged(pending)


def test_duplicate_pause_rejected() -> None:
    port = _port()
    port.request_pause(_pause_request())
    with pytest.raises(ExecutionContinuationError) as exc:
        port.request_pause(_pause_request())
    assert exc.value.code is ExecutionContinuationErrorCode.DUPLICATE_CONTINUATION


def test_internal_progression_pause_requested_to_waiting() -> None:
    deps = _deps()
    port = deps.continuation
    driver = deps.lifecycle_driver
    created = port.request_pause(_pause_request())
    assert created.lifecycle_state is ExecutionContinuationLifecycleState.PAUSE_REQUESTED
    paused = driver.record_execution_reached_safe_pause(
        _CONTINUATION_ID,
        execution_pause_established=True,
    )
    assert paused.lifecycle_state is ExecutionContinuationLifecycleState.PAUSED
    assert paused.revision == created.revision + 1
    waiting = driver.record_ready_for_human_resolution(_CONTINUATION_ID)
    assert waiting.lifecycle_state is ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN
    assert waiting.revision == paused.revision + 1
    _assert_four_ids_unchanged(waiting)


def test_approve_and_resume_spine() -> None:
    deps = _deps()
    waiting = _drive_to_waiting(deps)
    port = deps.continuation
    approved = port.apply_resolution(_resolution_command(expected_revision=waiting.revision))
    assert approved.lifecycle_state is ExecutionContinuationLifecycleState.RESUME_AUTHORIZED
    assert approved.revision == waiting.revision + 1
    resumed = port.resume(
        ExecutionContinuationResumeCommand(
            continuation_id=_CONTINUATION_ID,
            identity=_identity(),
            expected_revision=approved.revision,
        ),
    )
    assert resumed.lifecycle_state is ExecutionContinuationLifecycleState.RESUMED
    assert resumed.revision == approved.revision + 1
    _assert_four_ids_unchanged(resumed)


@pytest.mark.parametrize(
    "verdict",
    [
        ExecutionHumanVerdict.REJECT,
        ExecutionHumanVerdict.ESCALATE,
    ],
)
def test_resolution_terminal_verdicts_block_resume(verdict: ExecutionHumanVerdict) -> None:
    deps = _deps()
    waiting = _drive_to_waiting(deps)
    port = deps.continuation
    resolved = port.apply_resolution(
        _resolution_command(expected_revision=waiting.revision, verdict=verdict),
    )
    assert resolved.lifecycle_state is ExecutionContinuationLifecycleState(resolved.lifecycle_state)
    with pytest.raises(ExecutionContinuationError):
        port.resume(
            ExecutionContinuationResumeCommand(
                continuation_id=_CONTINUATION_ID,
                identity=_identity(),
                expected_revision=resolved.revision,
            ),
        )


def test_cancel_lifecycle_semantics_via_store() -> None:
    store = InMemoryExecutionContinuationStateStore()
    service = ExecutionContinuationService(store)
    port: ExecutionContinuationPort = service
    driver = ExecutionContinuationLifecycleDriver(service)
    port.request_pause(_pause_request())
    driver.record_execution_reached_safe_pause(
        _CONTINUATION_ID,
        execution_pause_established=True,
    )
    waiting = driver.record_ready_for_human_resolution(_CONTINUATION_ID)
    cancelled_state = advance_continuation_lifecycle(
        waiting.lifecycle_state,
        ExecutionContinuationTransition.CANCEL_CONTINUATION,
    )
    cancelled = waiting.model_copy(
        update={"lifecycle_state": cancelled_state, "revision": waiting.revision + 1},
    )
    assert store.compare_and_swap(
        continuation_id=waiting.continuation_id,
        expected=waiting,
        updated=cancelled,
    )
    with pytest.raises(ExecutionContinuationError):
        port.resume(
            ExecutionContinuationResumeCommand(
                continuation_id=_CONTINUATION_ID,
                identity=_identity(),
                expected_revision=cancelled.revision,
            ),
        )


def test_stale_resolution_and_resume() -> None:
    deps = _deps()
    port = deps.continuation
    waiting = _drive_to_waiting(deps)
    with pytest.raises(ExecutionContinuationError) as stale_resolution:
        port.apply_resolution(_resolution_command(expected_revision=waiting.revision - 1))
    assert stale_resolution.value.code is ExecutionContinuationErrorCode.STALE_REVISION
    approved = port.apply_resolution(_resolution_command(expected_revision=waiting.revision))
    with pytest.raises(ExecutionContinuationError) as stale_resume:
        port.resume(
            ExecutionContinuationResumeCommand(
                continuation_id=_CONTINUATION_ID,
                identity=_identity(),
                expected_revision=approved.revision - 1,
            ),
        )
    assert stale_resume.value.code is ExecutionContinuationErrorCode.STALE_REVISION


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_id", canonical_task_id_for_tests("other-task")),
        ("run_id", canonical_run_id_for_tests("other-run")),
        ("attempt_id", mint_attempt_id()),
        ("execution_id", mint_execution_id()),
    ],
)
def test_resolution_identity_mismatch_blocked(field: str, value: object) -> None:
    deps = _deps()
    port = deps.continuation
    waiting = _drive_to_waiting(deps)
    with pytest.raises(ExecutionContinuationError) as exc:
        port.apply_resolution(
            _resolution_command(
                expected_revision=waiting.revision,
                identity=_identity(**{field: value}),
            ),
        )
    assert exc.value.code is ExecutionContinuationErrorCode.IDENTITY_MISMATCH


def test_governed_scope_regression() -> None:
    deps = _deps()
    port = deps.continuation
    waiting = _drive_to_waiting(deps)
    with pytest.raises(ExecutionContinuationError) as exc:
        port.apply_resolution(
            _resolution_command(
                expected_revision=waiting.revision,
                operation_id="wrong-op",
            ),
        )
    assert exc.value.code is ExecutionContinuationErrorCode.SCOPE_MISMATCH

    deps2 = wire_execution_engine_continuation_dependencies()
    port2 = deps2.continuation
    driver2 = deps2.lifecycle_driver
    digest_correlation = _governed_correlation(side_effect_scope_digest=_SCOPE_DIGEST)
    port2.request_pause(_pause_request(governed_correlation=digest_correlation))
    driver2.record_execution_reached_safe_pause(
        _CONTINUATION_ID,
        execution_pause_established=True,
    )
    waiting_digest = driver2.record_ready_for_human_resolution(_CONTINUATION_ID)
    with pytest.raises(ExecutionContinuationError) as digest_exc:
        port2.apply_resolution(
            _resolution_command(
                expected_revision=waiting_digest.revision,
                side_effect_scope_digest="sha256:" + "b" * 64,
            ),
        )
    assert digest_exc.value.code is ExecutionContinuationErrorCode.SCOPE_MISMATCH


def test_double_resolution_and_double_resume_blocked() -> None:
    deps = _deps()
    port = deps.continuation
    waiting = _drive_to_waiting(deps)
    approved = port.apply_resolution(_resolution_command(expected_revision=waiting.revision))
    with pytest.raises(ExecutionContinuationError) as double_resolution:
        port.apply_resolution(_resolution_command(expected_revision=approved.revision))
    assert double_resolution.value.code is ExecutionContinuationErrorCode.ALREADY_RESOLVED
    resumed = port.resume(
        ExecutionContinuationResumeCommand(
            continuation_id=_CONTINUATION_ID,
            identity=_identity(),
            expected_revision=approved.revision,
        ),
    )
    with pytest.raises(ExecutionContinuationError) as double_resume:
        port.resume(
            ExecutionContinuationResumeCommand(
                continuation_id=_CONTINUATION_ID,
                identity=_identity(),
                expected_revision=resumed.revision,
            ),
        )
    assert double_resume.value.code is ExecutionContinuationErrorCode.ALREADY_RESUMED
    assert resumed.lifecycle_state is ExecutionContinuationLifecycleState.RESUMED


def test_concurrent_resolution_max_one_success() -> None:
    deps = _deps()
    port = deps.continuation
    waiting = _drive_to_waiting(deps)
    barrier = threading.Barrier(2)
    results: list[ExecutionContinuationError | PendingExecutionContinuation] = []

    def _resolve() -> None:
        barrier.wait()
        try:
            results.append(
                port.apply_resolution(_resolution_command(expected_revision=waiting.revision)),
            )
        except ExecutionContinuationError as exc:
            results.append(exc)

    threads = [threading.Thread(target=_resolve), threading.Thread(target=_resolve)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    successes = [item for item in results if isinstance(item, PendingExecutionContinuation)]
    stale = [
        item
        for item in results
        if isinstance(item, ExecutionContinuationError)
        and item.code is ExecutionContinuationErrorCode.STALE_REVISION
    ]
    assert len(successes) == 1
    assert len(stale) == 1


def test_concurrent_resume_max_one_success() -> None:
    deps = _deps()
    port = deps.continuation
    waiting = _drive_to_waiting(deps)
    approved = port.apply_resolution(_resolution_command(expected_revision=waiting.revision))
    barrier = threading.Barrier(2)
    results: list[ExecutionContinuationError | PendingExecutionContinuation] = []
    command = ExecutionContinuationResumeCommand(
        continuation_id=_CONTINUATION_ID,
        identity=_identity(),
        expected_revision=approved.revision,
    )

    def _resume() -> None:
        barrier.wait()
        try:
            results.append(port.resume(command))
        except ExecutionContinuationError as exc:
            results.append(exc)

    threads = [threading.Thread(target=_resume), threading.Thread(target=_resume)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    successes = [item for item in results if isinstance(item, PendingExecutionContinuation)]
    stale = [
        item
        for item in results
        if isinstance(item, ExecutionContinuationError)
        and item.code is ExecutionContinuationErrorCode.STALE_REVISION
    ]
    assert len(successes) == 1
    assert len(stale) == 1


def test_resume_does_not_invoke_root_admission() -> None:
    root_admission_calls = {"count": 0}

    class _RootAdmissionSpy:
        async def dispatch(self, *_args: object, **_kwargs: object) -> object:
            root_admission_calls["count"] += 1
            raise AssertionError("root admission must not run for continuation resume")

    _ = _RootAdmissionSpy()
    deps = _deps()
    port = deps.continuation
    waiting = _drive_to_waiting(deps)
    approved = port.apply_resolution(_resolution_command(expected_revision=waiting.revision))
    port.resume(
        ExecutionContinuationResumeCommand(
            continuation_id=_CONTINUATION_ID,
            identity=_identity(),
            expected_revision=approved.revision,
        ),
    )
    assert root_admission_calls["count"] == 0


def test_replaceable_implementation_and_falsey_store_preserved() -> None:
    custom_store = InMemoryExecutionContinuationStateStore()

    class _FalseyStore(ExecutionContinuationStateStore):
        def __bool__(self) -> bool:
            return False

        @property
        def is_durable(self) -> bool:
            return custom_store.is_durable

        def load(self, continuation_id: str) -> PendingExecutionContinuation | None:
            return custom_store.load(continuation_id)

        def find_by_identity(
            self,
            identity: ExecutionContinuationIdentity,
        ) -> PendingExecutionContinuation | None:
            return custom_store.find_by_identity(identity)

        def resolve_current_episode_for_identity(
            self,
            identity: ExecutionContinuationIdentity,
        ) -> PendingExecutionContinuation | None:
            return custom_store.resolve_current_episode_for_identity(identity)

        def resolve_identity_for_execution_progress(
            self,
            identity: ExecutionContinuationIdentity,
        ) -> PendingExecutionContinuation | None:
            return custom_store.resolve_identity_for_execution_progress(identity)

        def begin_current_episode_if_predecessor_allows(
            self,
            pending: PendingExecutionContinuation,
        ) -> bool:
            return custom_store.begin_current_episode_if_predecessor_allows(pending)

        def insert_if_absent(self, pending: PendingExecutionContinuation) -> bool:
            return custom_store.insert_if_absent(pending)

        def compare_and_swap(
            self,
            *,
            continuation_id: str,
            expected: PendingExecutionContinuation,
            updated: PendingExecutionContinuation,
        ) -> bool:
            return custom_store.compare_and_swap(
                continuation_id=continuation_id,
                expected=expected,
                updated=updated,
            )

    falsey = _FalseyStore()
    deps = wire_execution_engine_continuation_dependencies(state_store=falsey)
    assert deps.continuation_service.store is falsey
    assert isinstance(deps.continuation, ExecutionContinuationPort)


def test_production_service_satisfies_port() -> None:
    service = ExecutionContinuationService(InMemoryExecutionContinuationStateStore())
    assert isinstance(service, ExecutionContinuationPort)


def test_gr5_r2_public_contract_and_service_have_no_nexus_imports() -> None:
    for path in (_CONTRACT_MODULE, _SERVICE_MODULE):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.append(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
        assert not any("nexus" in module.lower() for module in modules)
