# © Artur Czarnecki. All rights reserved.

"""GR-5-R1 — ExecutionContinuationPort contract, state machine, and pluginability."""

from __future__ import annotations

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
    apply_resolution_to_pending,
    apply_resume_to_pending,
    assert_execution_continuation_identity_match,
    assert_pending_matches_resolution_command,
)
from intergrax.contracts.governed_continuation_correlation import (
    ContinuationReason,
    GovernedContinuationCorrelation,
)
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id
from testing_support.builder import canonical_run_id_for_tests, canonical_task_id_for_tests

_SEED = "gr5-r1-continuation"
_TASK = canonical_task_id_for_tests(_SEED)
_RUN = canonical_run_id_for_tests(_SEED)
_ATTEMPT = mint_attempt_id()
_EXECUTION = mint_execution_id()
_CONTINUATION_ID = "gcr_gr5_r1_test"


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
        "operation_id": "op_gr5_r1",
        "side_effect_scope_id": "scope_gr5_r1",
    }
    payload.update(overrides)
    return GovernedContinuationCorrelation.model_validate(payload)


def _waiting_pending(revision: int = 1) -> PendingExecutionContinuation:
    state = ExecutionContinuationLifecycleState.PAUSE_REQUESTED
    for transition in (
        ExecutionContinuationTransition.ADVANCE_TO_PAUSED,
        ExecutionContinuationTransition.ADVANCE_TO_HUMAN_WAIT,
    ):
        state = advance_continuation_lifecycle(state, transition)
    return PendingExecutionContinuation(
        continuation_id=_CONTINUATION_ID,
        identity=_identity(),
        lifecycle_state=state,
        revision=revision,
        reason=ContinuationReason.SECURITY,
        governed_correlation=_governed_correlation(),
        pause_id="pause_gr5",
        human_request_id="hr_gr5",
        requested_at="2026-09-15T00:00:00Z",
    )


def _resolution_command(
    *,
    expected_revision: int = 1,
    verdict: ExecutionHumanVerdict = ExecutionHumanVerdict.APPROVE,
    **overrides: object,
) -> ExecutionContinuationResolutionCommand:
    payload: dict[str, object] = {
        "continuation_id": _CONTINUATION_ID,
        "identity": _identity(),
        "expected_revision": expected_revision,
        "verdict": verdict,
        "approver": local_development_approver_evidence(
            actor_id="op-1",
            tenant_id="tenant-gr5",
        ),
        "human_request_id": "hr_gr5",
        "pause_id": "pause_gr5",
        "operation_id": "op_gr5_r1",
        "side_effect_scope_id": "scope_gr5_r1",
        "resolved_at": "2026-09-15T01:00:00Z",
    }
    payload.update(overrides)
    return ExecutionContinuationResolutionCommand.model_validate(payload)


class _DictExecutionContinuationPort:
    """Test double A — dict keyed by continuation_id."""

    def __init__(self) -> None:
        self._store: dict[str, PendingExecutionContinuation] = {}

    def request_pause(self, request: ExecutionPauseRequest) -> PendingExecutionContinuation:
        if request.continuation_id in self._store:
            raise ExecutionContinuationError(
                "duplicate continuation",
                code=ExecutionContinuationErrorCode.DUPLICATE_CONTINUATION,
            )
        state = advance_continuation_lifecycle(
            None,
            ExecutionContinuationTransition.REQUEST_PAUSE,
        )
        pending = PendingExecutionContinuation(
            continuation_id=request.continuation_id,
            identity=request.identity,
            lifecycle_state=state,
            revision=1,
            reason=request.reason,
            governed_correlation=request.governed_correlation,
            pause_id=request.pause_id,
            human_request_id=request.human_request_id,
            requested_at=request.requested_at,
        )
        self._store[request.continuation_id] = pending
        return pending

    def _advance_to_human_wait(self, continuation_id: str) -> PendingExecutionContinuation:
        pending = self._require(continuation_id)
        state = advance_continuation_lifecycle(
            pending.lifecycle_state,
            ExecutionContinuationTransition.ADVANCE_TO_PAUSED,
        )
        state = advance_continuation_lifecycle(
            state,
            ExecutionContinuationTransition.ADVANCE_TO_HUMAN_WAIT,
        )
        updated = pending.model_copy(
            update={"lifecycle_state": state, "revision": pending.revision + 1},
        )
        self._store[continuation_id] = updated
        return updated

    def _require(self, continuation_id: str) -> PendingExecutionContinuation:
        try:
            return self._store[continuation_id]
        except KeyError:
            raise ExecutionContinuationError(
                "continuation not found",
                code=ExecutionContinuationErrorCode.NOT_FOUND,
            )

    def get_pending(self, lookup: ExecutionContinuationLookup) -> PendingExecutionContinuation:
        if lookup.continuation_id is not None:
            pending = self._require(lookup.continuation_id)
        else:
            matches = [
                p
                for p in self._store.values()
                if lookup.identity is not None
                and p.identity.task_id == lookup.identity.task_id
                and p.identity.run_id == lookup.identity.run_id
                and p.identity.attempt_id == lookup.identity.attempt_id
                and p.identity.execution_id == lookup.identity.execution_id
            ]
            if len(matches) != 1:
                raise ExecutionContinuationError(
                    "continuation not found",
                    code=ExecutionContinuationErrorCode.NOT_FOUND,
                )
            pending = matches[0]
        if lookup.identity is not None:
            assert_execution_continuation_identity_match(lookup.identity, pending.identity)
        return pending

    def apply_resolution(
        self,
        command: ExecutionContinuationResolutionCommand,
    ) -> PendingExecutionContinuation:
        pending = self._require(command.continuation_id)
        if pending.lifecycle_state is ExecutionContinuationLifecycleState.PAUSE_REQUESTED:
            pending = self._advance_to_human_wait(command.continuation_id)
        updated = apply_resolution_to_pending(pending, command)
        self._store[command.continuation_id] = updated
        return updated

    def resume(self, command: ExecutionContinuationResumeCommand) -> PendingExecutionContinuation:
        pending = self._require(command.continuation_id)
        updated = apply_resume_to_pending(pending, command)
        self._store[command.continuation_id] = updated
        return updated


class _ListExecutionContinuationPort:
    """Test double B — linear store (different structure, same semantics)."""

    def __init__(self) -> None:
        self._entries: list[PendingExecutionContinuation] = []

    def request_pause(self, request: ExecutionPauseRequest) -> PendingExecutionContinuation:
        if any(e.continuation_id == request.continuation_id for e in self._entries):
            raise ExecutionContinuationError(
                "duplicate continuation",
                code=ExecutionContinuationErrorCode.DUPLICATE_CONTINUATION,
            )
        state = advance_continuation_lifecycle(
            None,
            ExecutionContinuationTransition.REQUEST_PAUSE,
        )
        pending = PendingExecutionContinuation(
            continuation_id=request.continuation_id,
            identity=request.identity,
            lifecycle_state=state,
            revision=1,
            reason=request.reason,
            governed_correlation=request.governed_correlation,
        )
        self._entries.append(pending)
        return pending

    def _find(self, continuation_id: str) -> PendingExecutionContinuation:
        for entry in self._entries:
            if entry.continuation_id == continuation_id:
                return entry
        raise ExecutionContinuationError(
            "continuation not found",
            code=ExecutionContinuationErrorCode.NOT_FOUND,
        )

    def _replace(self, updated: PendingExecutionContinuation) -> None:
        for index, entry in enumerate(self._entries):
            if entry.continuation_id == updated.continuation_id:
                self._entries[index] = updated
                return
        self._entries.append(updated)

    def get_pending(self, lookup: ExecutionContinuationLookup) -> PendingExecutionContinuation:
        if lookup.continuation_id is not None:
            pending = self._find(lookup.continuation_id)
        elif lookup.identity is not None:
            matches = [e for e in self._entries if e.identity == lookup.identity]
            if len(matches) != 1:
                raise ExecutionContinuationError(
                    "continuation not found",
                    code=ExecutionContinuationErrorCode.NOT_FOUND,
                )
            pending = matches[0]
        else:
            raise ExecutionContinuationError(
                "continuation not found",
                code=ExecutionContinuationErrorCode.NOT_FOUND,
            )
        if lookup.identity is not None:
            assert_execution_continuation_identity_match(lookup.identity, pending.identity)
        return pending

    def _advance_to_human_wait(self, continuation_id: str) -> PendingExecutionContinuation:
        pending = self._find(continuation_id)
        state = advance_continuation_lifecycle(
            pending.lifecycle_state,
            ExecutionContinuationTransition.ADVANCE_TO_PAUSED,
        )
        state = advance_continuation_lifecycle(
            state,
            ExecutionContinuationTransition.ADVANCE_TO_HUMAN_WAIT,
        )
        updated = pending.model_copy(
            update={"lifecycle_state": state, "revision": pending.revision + 1},
        )
        self._replace(updated)
        return updated

    def apply_resolution(
        self,
        command: ExecutionContinuationResolutionCommand,
    ) -> PendingExecutionContinuation:
        pending = self._find(command.continuation_id)
        if pending.lifecycle_state is ExecutionContinuationLifecycleState.PAUSE_REQUESTED:
            pending = self._advance_to_human_wait(command.continuation_id)
        updated = apply_resolution_to_pending(pending, command)
        self._replace(updated)
        return updated

    def resume(self, command: ExecutionContinuationResumeCommand) -> PendingExecutionContinuation:
        pending = self._find(command.continuation_id)
        updated = apply_resume_to_pending(pending, command)
        self._replace(updated)
        return updated


def _drive_approve_and_resume(port: ExecutionContinuationPort) -> PendingExecutionContinuation:
    port.request_pause(
        ExecutionPauseRequest(
            identity=_identity(),
            continuation_id=_CONTINUATION_ID,
            reason=ContinuationReason.SECURITY,
            governed_correlation=_governed_correlation(),
        ),
    )
    if isinstance(port, (_DictExecutionContinuationPort, _ListExecutionContinuationPort)):
        waiting = port._advance_to_human_wait(_CONTINUATION_ID)
    else:
        waiting = port.get_pending(ExecutionContinuationLookup(continuation_id=_CONTINUATION_ID))
    approved = port.apply_resolution(
        _resolution_command(expected_revision=waiting.revision),
    )
    assert approved.lifecycle_state is ExecutionContinuationLifecycleState.RESUME_AUTHORIZED
    resumed = port.resume(
        ExecutionContinuationResumeCommand(
            continuation_id=_CONTINUATION_ID,
            identity=_identity(),
            expected_revision=approved.revision,
        ),
    )
    assert resumed.lifecycle_state is ExecutionContinuationLifecycleState.RESUMED
    return resumed


@pytest.mark.parametrize(
    "factory",
    [_DictExecutionContinuationPort, _ListExecutionContinuationPort],
)
def test_pluginability_two_implementations(factory: type[ExecutionContinuationPort]) -> None:
    port = factory()
    assert isinstance(port, ExecutionContinuationPort)
    resumed = _drive_approve_and_resume(port)
    assert resumed.revision >= 3
    restored = PendingExecutionContinuation.model_validate_json(resumed.model_dump_json())
    assert restored.identity.execution_id == _EXECUTION


def test_state_machine_valid_spine() -> None:
    state = advance_continuation_lifecycle(
        None,
        ExecutionContinuationTransition.REQUEST_PAUSE,
    )
    state = advance_continuation_lifecycle(
        state,
        ExecutionContinuationTransition.ADVANCE_TO_PAUSED,
    )
    state = advance_continuation_lifecycle(
        state,
        ExecutionContinuationTransition.ADVANCE_TO_HUMAN_WAIT,
    )
    state = advance_continuation_lifecycle(
        state,
        ExecutionContinuationTransition.APPLY_RESOLUTION,
        verdict=ExecutionHumanVerdict.APPROVE,
    )
    assert state is ExecutionContinuationLifecycleState.RESUME_AUTHORIZED
    state = advance_continuation_lifecycle(state, ExecutionContinuationTransition.RESUME)
    assert state is ExecutionContinuationLifecycleState.RESUMED


def test_invalid_transition_resumed_to_waiting() -> None:
    with pytest.raises(ExecutionContinuationError) as exc:
        advance_continuation_lifecycle(
            ExecutionContinuationLifecycleState.RESUMED,
            ExecutionContinuationTransition.ADVANCE_TO_HUMAN_WAIT,
        )
    assert exc.value.code is ExecutionContinuationErrorCode.INVALID_TRANSITION


def test_four_id_mismatch_matrix() -> None:
    base = _identity()
    other_task = canonical_task_id_for_tests("other-task")
    other_run = canonical_run_id_for_tests("other-run")
    other_attempt = mint_attempt_id()
    other_execution = mint_execution_id()
    for field, wrong in (
        ("task_id", other_task),
        ("run_id", other_run),
        ("attempt_id", other_attempt),
        ("execution_id", other_execution),
    ):
        kwargs = {
            "task_id": base.task_id,
            "run_id": base.run_id,
            "attempt_id": base.attempt_id,
            "execution_id": base.execution_id,
            field: wrong,
        }
        with pytest.raises(ExecutionContinuationError) as exc:
            assert_execution_continuation_identity_match(base, ExecutionContinuationIdentity(**kwargs))
        assert exc.value.code is ExecutionContinuationErrorCode.IDENTITY_MISMATCH


def test_stale_revision_on_resolution() -> None:
    pending = _waiting_pending(revision=2)
    command = _resolution_command(expected_revision=1)
    with pytest.raises(ExecutionContinuationError) as exc:
        assert_pending_matches_resolution_command(pending, command)
    assert exc.value.code is ExecutionContinuationErrorCode.STALE_REVISION


def test_wrong_continuation_id_blocks() -> None:
    pending = _waiting_pending()
    command = _resolution_command(continuation_id="gcr_wrong")
    with pytest.raises(ExecutionContinuationError) as exc:
        assert_pending_matches_resolution_command(pending, command)
    assert exc.value.code is ExecutionContinuationErrorCode.IDENTITY_MISMATCH


def test_scope_mismatch_blocks() -> None:
    pending = _waiting_pending()
    command = _resolution_command(operation_id="op_other")
    with pytest.raises(ExecutionContinuationError) as exc:
        assert_pending_matches_resolution_command(pending, command)
    assert exc.value.code is ExecutionContinuationErrorCode.SCOPE_MISMATCH


def test_duplicate_resolution_semantic() -> None:
    pending = _waiting_pending()
    approved = apply_resolution_to_pending(pending, _resolution_command())
    with pytest.raises(ExecutionContinuationError) as exc:
        apply_resolution_to_pending(approved, _resolution_command(expected_revision=2))
    assert exc.value.code is ExecutionContinuationErrorCode.ALREADY_RESOLVED


def test_duplicate_resume_semantic() -> None:
    pending = _waiting_pending()
    authorized = apply_resolution_to_pending(pending, _resolution_command())
    resumed = apply_resume_to_pending(
        authorized,
        ExecutionContinuationResumeCommand(
            continuation_id=_CONTINUATION_ID,
            identity=_identity(),
            expected_revision=authorized.revision,
        ),
    )
    with pytest.raises(ExecutionContinuationError) as exc:
        apply_resume_to_pending(
            resumed,
            ExecutionContinuationResumeCommand(
                continuation_id=_CONTINUATION_ID,
                identity=_identity(),
                expected_revision=resumed.revision,
            ),
        )
    assert exc.value.code is ExecutionContinuationErrorCode.ALREADY_RESUMED


def test_lookup_requires_full_key() -> None:
    with pytest.raises(ValueError, match="continuation_id or full identity"):
        ExecutionContinuationLookup()


def test_governed_continuation_regression_import() -> None:
    from intergrax.contracts.governed_continuation import GovernedContinuationRequest

    req = GovernedContinuationRequest(
        reason=ContinuationReason.QUOTE,
        task_id=_TASK,
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXECUTION,
        source_agent_id="agent",
        prompt="prompt",
        continuation_request_id=_CONTINUATION_ID,
    )
    assert req.continuation_request_id == _CONTINUATION_ID
