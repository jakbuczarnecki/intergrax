# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision System lifecycle stage contracts (DS-CORE-03).

Declarative semantic progression model hosted by canonical Execution
— not a second runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from intergrax.contracts.decision_identity import DecisionIdentity


class DecisionLifecycleStage(str, Enum):
    """Canonical semantic stages for one decision lifecycle."""

    PROPOSAL = "proposal"
    DELIBERATION = "deliberation"
    VERIFICATION = "verification"
    REVISION = "revision"
    ADJUDICATION = "adjudication"
    RESOLUTION = "resolution"
    FINALIZATION = "finalization"
    TERMINAL = "terminal"


_ALLOWED_TRANSITIONS: dict[
    DecisionLifecycleStage,
    frozenset[DecisionLifecycleStage],
] = {
    DecisionLifecycleStage.PROPOSAL: frozenset(
        {
            DecisionLifecycleStage.DELIBERATION,
            DecisionLifecycleStage.VERIFICATION,
        },
    ),
    DecisionLifecycleStage.DELIBERATION: frozenset(
        {DecisionLifecycleStage.VERIFICATION},
    ),
    DecisionLifecycleStage.VERIFICATION: frozenset(
        {
            DecisionLifecycleStage.REVISION,
            DecisionLifecycleStage.ADJUDICATION,
            DecisionLifecycleStage.RESOLUTION,
        },
    ),
    DecisionLifecycleStage.REVISION: frozenset(
        {DecisionLifecycleStage.VERIFICATION},
    ),
    DecisionLifecycleStage.ADJUDICATION: frozenset(
        {
            DecisionLifecycleStage.REVISION,
            DecisionLifecycleStage.RESOLUTION,
        },
    ),
    DecisionLifecycleStage.RESOLUTION: frozenset(
        {DecisionLifecycleStage.FINALIZATION},
    ),
    DecisionLifecycleStage.FINALIZATION: frozenset(
        {DecisionLifecycleStage.TERMINAL},
    ),
    DecisionLifecycleStage.TERMINAL: frozenset(),
}


@dataclass(frozen=True, slots=True)
class DecisionLifecycleTransition:
    """Explicit allowed forward transition between lifecycle stages."""

    from_stage: DecisionLifecycleStage
    to_stage: DecisionLifecycleStage

    def __post_init__(self) -> None:
        if type(self.from_stage) is not DecisionLifecycleStage:
            raise TypeError(
                "DecisionLifecycleTransition.from_stage must be DecisionLifecycleStage",
            )
        if type(self.to_stage) is not DecisionLifecycleStage:
            raise TypeError(
                "DecisionLifecycleTransition.to_stage must be DecisionLifecycleStage",
            )


@dataclass(frozen=True, slots=True)
class DecisionLifecycleState:
    """Immutable lifecycle position for one decision identity."""

    identity: DecisionIdentity
    stage: DecisionLifecycleStage
    transition_index: int

    def __post_init__(self) -> None:
        if type(self.identity) is not DecisionIdentity:
            raise TypeError("DecisionLifecycleState.identity must be DecisionIdentity")
        if type(self.stage) is not DecisionLifecycleStage:
            raise TypeError("DecisionLifecycleState.stage must be DecisionLifecycleStage")
        if type(self.transition_index) is not int or isinstance(self.transition_index, bool):
            raise TypeError("DecisionLifecycleState.transition_index must be int")
        if self.transition_index < 0:
            raise ValueError("DecisionLifecycleState.transition_index must be >= 0")


def initial_decision_lifecycle_state(
    identity: DecisionIdentity,
) -> DecisionLifecycleState:
    """Return the canonical initial lifecycle state for one decision identity."""
    if type(identity) is not DecisionIdentity:
        raise TypeError("identity must be DecisionIdentity")
    return DecisionLifecycleState(
        identity=identity,
        stage=DecisionLifecycleStage.PROPOSAL,
        transition_index=0,
    )


def validate_lifecycle_transition(
    *,
    from_stage: DecisionLifecycleStage,
    to_stage: DecisionLifecycleStage,
) -> DecisionLifecycleTransition:
    """Validate an architecturally legal lifecycle transition."""
    if type(from_stage) is not DecisionLifecycleStage:
        raise TypeError("from_stage must be DecisionLifecycleStage")
    if type(to_stage) is not DecisionLifecycleStage:
        raise TypeError("to_stage must be DecisionLifecycleStage")
    if from_stage == to_stage:
        raise ValueError(
            f"Unsupported lifecycle transition: {from_stage.value} -> {to_stage.value}",
        )
    allowed = _ALLOWED_TRANSITIONS.get(from_stage, frozenset())
    if to_stage not in allowed:
        raise ValueError(
            f"Unsupported lifecycle transition: {from_stage.value} -> {to_stage.value}",
        )
    return DecisionLifecycleTransition(from_stage=from_stage, to_stage=to_stage)


def transition_decision_lifecycle(
    state: DecisionLifecycleState,
    to_stage: DecisionLifecycleStage,
) -> DecisionLifecycleState:
    """Apply one validated semantic transition and return a new immutable state."""
    if type(state) is not DecisionLifecycleState:
        raise TypeError("state must be DecisionLifecycleState")
    validate_lifecycle_transition(from_stage=state.stage, to_stage=to_stage)
    return DecisionLifecycleState(
        identity=state.identity,
        stage=to_stage,
        transition_index=state.transition_index + 1,
    )
