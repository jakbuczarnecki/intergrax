# © Artur Czarnecki. All rights reserved.

from dataclasses import FrozenInstanceError

import pytest

from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_lifecycle import (
    DecisionLifecycleStage,
    DecisionLifecycleState,
    DecisionLifecycleTransition,
    initial_decision_lifecycle_state,
    transition_decision_lifecycle,
    validate_lifecycle_transition,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)

_CANONICAL_STAGES = (
    DecisionLifecycleStage.PROPOSAL,
    DecisionLifecycleStage.DELIBERATION,
    DecisionLifecycleStage.VERIFICATION,
    DecisionLifecycleStage.REVISION,
    DecisionLifecycleStage.ADJUDICATION,
    DecisionLifecycleStage.RESOLUTION,
    DecisionLifecycleStage.FINALIZATION,
    DecisionLifecycleStage.TERMINAL,
)

_FORBIDDEN_EXECUTION_STAGE_NAMES = frozenset(
    {
        "RETRY",
        "RETRYING",
        "FAILED",
        "CANCELLED",
        "CANCELED",
        "TIMED_OUT",
        "PAUSED",
        "RUNNING",
        "COMPLETED",
        "WAITING_FOR_HUMAN",
        "HITL_PENDING",
    },
)


def _lineage() -> DecisionExecutionLineage:
    return DecisionExecutionLineage(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _identity() -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="incident", subject="incident-123"),
        tenant_id="tenant-a",
        execution=_lineage(),
    )


def _advance(
    state: DecisionLifecycleState,
    *stages: DecisionLifecycleStage,
) -> DecisionLifecycleState:
    current = state
    for stage in stages:
        current = transition_decision_lifecycle(current, stage)
    return current


@pytest.mark.unit
@pytest.mark.gate
def test_canonical_lifecycle_stages_exact_set() -> None:
    assert tuple(DecisionLifecycleStage) == _CANONICAL_STAGES


@pytest.mark.unit
@pytest.mark.gate
def test_lifecycle_stages_exclude_execution_semantics() -> None:
    member_names = frozenset(stage.name for stage in DecisionLifecycleStage)
    assert member_names.isdisjoint(_FORBIDDEN_EXECUTION_STAGE_NAMES)


@pytest.mark.unit
@pytest.mark.gate
def test_initial_decision_lifecycle_state() -> None:
    identity = _identity()
    state = initial_decision_lifecycle_state(identity)
    assert state.identity is identity
    assert state.stage is DecisionLifecycleStage.PROPOSAL
    assert state.transition_index == 0


@pytest.mark.unit
@pytest.mark.gate
def test_minimal_happy_path() -> None:
    identity = _identity()
    state = initial_decision_lifecycle_state(identity)
    path = (
        DecisionLifecycleStage.VERIFICATION,
        DecisionLifecycleStage.RESOLUTION,
        DecisionLifecycleStage.FINALIZATION,
        DecisionLifecycleStage.TERMINAL,
    )
    for index, stage in enumerate(path, start=1):
        previous = state
        state = transition_decision_lifecycle(state, stage)
        assert state is not previous
        assert state.identity is identity
        assert state.stage is stage
        assert state.transition_index == index


@pytest.mark.unit
@pytest.mark.gate
def test_full_optional_path() -> None:
    identity = _identity()
    state = initial_decision_lifecycle_state(identity)
    state = _advance(
        state,
        DecisionLifecycleStage.DELIBERATION,
        DecisionLifecycleStage.VERIFICATION,
        DecisionLifecycleStage.REVISION,
        DecisionLifecycleStage.VERIFICATION,
        DecisionLifecycleStage.ADJUDICATION,
        DecisionLifecycleStage.RESOLUTION,
        DecisionLifecycleStage.FINALIZATION,
        DecisionLifecycleStage.TERMINAL,
    )
    assert state.identity is identity
    assert state.stage is DecisionLifecycleStage.TERMINAL
    assert state.transition_index == 8


@pytest.mark.unit
@pytest.mark.gate
def test_revision_loop_is_legal() -> None:
    identity = _identity()
    state = initial_decision_lifecycle_state(identity)
    state = _advance(state, DecisionLifecycleStage.VERIFICATION)
    state = _advance(
        state,
        DecisionLifecycleStage.REVISION,
        DecisionLifecycleStage.VERIFICATION,
        DecisionLifecycleStage.REVISION,
        DecisionLifecycleStage.VERIFICATION,
    )
    assert state.identity is identity
    assert state.stage is DecisionLifecycleStage.VERIFICATION
    assert state.transition_index == 5


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize(
    ("from_stage", "to_stage"),
    [
        (DecisionLifecycleStage.PROPOSAL, DecisionLifecycleStage.REVISION),
        (DecisionLifecycleStage.PROPOSAL, DecisionLifecycleStage.RESOLUTION),
        (DecisionLifecycleStage.DELIBERATION, DecisionLifecycleStage.REVISION),
        (DecisionLifecycleStage.REVISION, DecisionLifecycleStage.RESOLUTION),
        (DecisionLifecycleStage.RESOLUTION, DecisionLifecycleStage.VERIFICATION),
        (DecisionLifecycleStage.FINALIZATION, DecisionLifecycleStage.PROPOSAL),
        (DecisionLifecycleStage.TERMINAL, DecisionLifecycleStage.PROPOSAL),
        (DecisionLifecycleStage.TERMINAL, DecisionLifecycleStage.TERMINAL),
    ],
)
def test_invalid_transitions_raise_value_error(
    from_stage: DecisionLifecycleStage,
    to_stage: DecisionLifecycleStage,
) -> None:
    with pytest.raises(ValueError, match="Unsupported lifecycle transition"):
        validate_lifecycle_transition(from_stage=from_stage, to_stage=to_stage)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("stage", list(DecisionLifecycleStage))
def test_self_transition_is_invalid(stage: DecisionLifecycleStage) -> None:
    with pytest.raises(ValueError, match="Unsupported lifecycle transition"):
        validate_lifecycle_transition(from_stage=stage, to_stage=stage)


@pytest.mark.unit
@pytest.mark.gate
def test_state_and_transition_are_immutable() -> None:
    identity = _identity()
    state = initial_decision_lifecycle_state(identity)
    transition = validate_lifecycle_transition(
        from_stage=DecisionLifecycleStage.PROPOSAL,
        to_stage=DecisionLifecycleStage.VERIFICATION,
    )
    with pytest.raises(FrozenInstanceError):
        setattr(state, "stage", DecisionLifecycleStage.VERIFICATION)
    with pytest.raises(FrozenInstanceError):
        setattr(transition, "to_stage", DecisionLifecycleStage.TERMINAL)


@pytest.mark.unit
@pytest.mark.gate
def test_transition_does_not_mutate_original_state() -> None:
    identity = _identity()
    original = initial_decision_lifecycle_state(identity)
    updated = transition_decision_lifecycle(
        original,
        DecisionLifecycleStage.VERIFICATION,
    )
    assert original.stage is DecisionLifecycleStage.PROPOSAL
    assert original.transition_index == 0
    assert updated.stage is DecisionLifecycleStage.VERIFICATION
    assert updated.transition_index == 1


@pytest.mark.unit
@pytest.mark.gate
def test_identity_preserved_across_full_path() -> None:
    identity = _identity()
    state = initial_decision_lifecycle_state(identity)
    final_state = _advance(
        state,
        DecisionLifecycleStage.DELIBERATION,
        DecisionLifecycleStage.VERIFICATION,
        DecisionLifecycleStage.REVISION,
        DecisionLifecycleStage.VERIFICATION,
        DecisionLifecycleStage.ADJUDICATION,
        DecisionLifecycleStage.RESOLUTION,
        DecisionLifecycleStage.FINALIZATION,
        DecisionLifecycleStage.TERMINAL,
    )
    assert final_state.identity is identity
