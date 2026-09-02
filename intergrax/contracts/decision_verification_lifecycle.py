# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision Verification → Lifecycle handoff contracts (DS-VER-PIPE-05).

Typed immutable boundary where Verification presents an exact ``VerificationResult``
to Decision Lifecycle without selecting or performing the next lifecycle stage.

``DecisionLifecycleState`` binds ``DecisionIdentity`` only — it does not carry
``DecisionLineageRef.branch_id``. The handoff therefore preserves the exact
``VerificationResult.proposal_ref`` (including branch identity) while aligning
lifecycle and verification on the shared ``DecisionIdentity`` envelope. Lifecycle
cannot independently reject a sibling-branch mismatch until branch context is
added to the lifecycle seam in a future Decision core change.
"""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.decision_identity import DecisionIdentity
from intergrax.contracts.decision_lifecycle import (
    DecisionLifecycleStage,
    DecisionLifecycleState,
)
from intergrax.contracts.decision_verification import (
    VerificationResult,
    validate_verification_result,
)


@dataclass(frozen=True, slots=True)
class DecisionVerificationHandoff:
    """Immutable typed seam: Lifecycle owns progression; Verification owns detection.

    ``lifecycle_state`` proves which decision identity Lifecycle currently processes.
    ``verification_result`` proves the exact evaluated proposal ref and outcome.
    Neither field is mutated by handoff construction.
    """

    lifecycle_state: DecisionLifecycleState
    verification_result: VerificationResult

    def __post_init__(self) -> None:
        if type(self.lifecycle_state) is not DecisionLifecycleState:
            raise TypeError(
                "DecisionVerificationHandoff.lifecycle_state must be DecisionLifecycleState",
            )
        if type(self.verification_result) is not VerificationResult:
            raise TypeError(
                "DecisionVerificationHandoff.verification_result must be VerificationResult",
            )
        _validate_handoff_invariants(self.lifecycle_state, self.verification_result)


def _validate_verification_lifecycle_stage(state: DecisionLifecycleState) -> None:
    if state.stage is not DecisionLifecycleStage.VERIFICATION:
        raise ValueError(
            "verification handoff requires lifecycle stage verification, "
            f"got {state.stage.value!r}",
        )


def _validate_identity_alignment(
    *,
    lifecycle_identity: DecisionIdentity,
    verification_identity: DecisionIdentity,
) -> None:
    if lifecycle_identity != verification_identity:
        raise ValueError(
            "verification handoff requires lifecycle identity to match "
            "VerificationResult.proposal_ref.identity",
        )


def _validate_handoff_invariants(
    state: DecisionLifecycleState,
    result: VerificationResult,
) -> None:
    _validate_verification_lifecycle_stage(state)
    _validate_identity_alignment(
        lifecycle_identity=state.identity,
        verification_identity=result.proposal_ref.identity,
    )


def handoff_verification_result(
    *,
    state: DecisionLifecycleState,
    result: VerificationResult,
) -> DecisionVerificationHandoff:
    """Present one exact verification result to Lifecycle without transitioning stage.

    Validates lifecycle stage ``VERIFICATION`` and full ``DecisionIdentity`` alignment.
    Does not mutate ``state``, ``result``, or select the next lifecycle stage.
    """
    if type(state) is not DecisionLifecycleState:
        raise TypeError("state must be DecisionLifecycleState")
    if type(result) is not VerificationResult:
        raise TypeError("result must be VerificationResult")
    return DecisionVerificationHandoff(
        lifecycle_state=state,
        verification_result=result,
    )


def validate_decision_verification_handoff(
    handoff: DecisionVerificationHandoff,
) -> DecisionVerificationHandoff:
    """Re-validate one handoff by reconstructing the canonical contract."""
    if type(handoff) is not DecisionVerificationHandoff:
        raise TypeError("handoff must be DecisionVerificationHandoff")
    validate_verification_result(handoff.verification_result)
    return DecisionVerificationHandoff(
        lifecycle_state=handoff.lifecycle_state,
        verification_result=handoff.verification_result,
    )
