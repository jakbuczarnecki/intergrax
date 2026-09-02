# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision revision policy contracts (DS-REV-01).

Canonical Decision-owned semantics for reacting to Verification challenges.
Revision authorization is distinct from verification, execution retry, and HITL.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from intergrax.contracts.decision_record import (
    DecisionProposalRef,
    decision_proposal_ref_sort_key,
)
from intergrax.contracts.decision_verification import (
    VerificationDisposition,
    VerificationResult,
)


class DecisionRevisionDisposition(str, Enum):
    """Semantic revision eligibility for one challenged proposal."""

    ALLOWED = "allowed"
    NOT_REQUIRED = "not_required"
    EXHAUSTED = "exhausted"


@dataclass(frozen=True, slots=True)
class DecisionRevisionPolicy:
    """Immutable revision budget configuration."""

    max_revisions: int

    def __post_init__(self) -> None:
        if type(self.max_revisions) is not int or isinstance(self.max_revisions, bool):
            raise TypeError("DecisionRevisionPolicy.max_revisions must be int")
        if self.max_revisions < 0:
            raise ValueError("DecisionRevisionPolicy.max_revisions must be >= 0")


def decision_revision_policy(*, max_revisions: int) -> DecisionRevisionPolicy:
    """Build one revision policy with explicit non-negative budget."""
    return DecisionRevisionPolicy(max_revisions=max_revisions)


@dataclass(frozen=True, slots=True)
class DecisionRevisionState:
    """Current semantic revision progress for one decision lineage."""

    revision_count: int

    def __post_init__(self) -> None:
        if type(self.revision_count) is not int or isinstance(self.revision_count, bool):
            raise TypeError("DecisionRevisionState.revision_count must be int")
        if self.revision_count < 0:
            raise ValueError("DecisionRevisionState.revision_count must be >= 0")


def initial_decision_revision_state() -> DecisionRevisionState:
    """Return revision state for an initial candidate (revision_count = 0)."""
    return DecisionRevisionState(revision_count=0)


@dataclass(frozen=True, slots=True)
class DecisionRevisionDecision:
    """Revision eligibility outcome bound to one exact challenged proposal."""

    disposition: DecisionRevisionDisposition
    proposal_ref: DecisionProposalRef
    revision_number: int | None = None

    def __post_init__(self) -> None:
        if type(self.disposition) is not DecisionRevisionDisposition:
            raise TypeError(
                "DecisionRevisionDecision.disposition must be DecisionRevisionDisposition",
            )
        if type(self.proposal_ref) is not DecisionProposalRef:
            raise TypeError(
                "DecisionRevisionDecision.proposal_ref must be DecisionProposalRef",
            )
        if self.revision_number is not None:
            if type(self.revision_number) is not int or isinstance(
                self.revision_number,
                bool,
            ):
                raise TypeError("DecisionRevisionDecision.revision_number must be int")
            if self.revision_number < 1:
                raise ValueError(
                    "DecisionRevisionDecision.revision_number must be >= 1 when set",
                )
        if self.disposition is DecisionRevisionDisposition.ALLOWED:
            if self.revision_number is None:
                raise ValueError(
                    "DecisionRevisionDecision.revision_number is required when ALLOWED",
                )
        elif self.revision_number is not None:
            raise ValueError(
                "DecisionRevisionDecision.revision_number must be None unless ALLOWED",
            )


@dataclass(frozen=True, slots=True)
class DecisionRevisionAuthorization:
    """Immutable authorization to mint one semantic revision for one proposal."""

    proposal_ref: DecisionProposalRef
    policy: DecisionRevisionPolicy
    revision_number: int

    def __post_init__(self) -> None:
        if type(self.proposal_ref) is not DecisionProposalRef:
            raise TypeError(
                "DecisionRevisionAuthorization.proposal_ref must be DecisionProposalRef",
            )
        if type(self.policy) is not DecisionRevisionPolicy:
            raise TypeError(
                "DecisionRevisionAuthorization.policy must be DecisionRevisionPolicy",
            )
        if type(self.revision_number) is not int or isinstance(self.revision_number, bool):
            raise TypeError(
                "DecisionRevisionAuthorization.revision_number must be int",
            )
        if self.revision_number < 1:
            raise ValueError(
                "DecisionRevisionAuthorization.revision_number must be >= 1",
            )


def _proposal_ref_key(ref: DecisionProposalRef) -> tuple[str | int | None, ...]:
    return decision_proposal_ref_sort_key(ref)


def proposal_refs_match(
    left: DecisionProposalRef,
    right: DecisionProposalRef,
) -> bool:
    """Return whether two proposal references denote the same exact version."""
    return _proposal_ref_key(left) == _proposal_ref_key(right)


@runtime_checkable
class DecisionRevisionPolicyEvaluator(Protocol):
    """Optional substitution seam for custom revision policy behavior."""

    def evaluate(
        self,
        *,
        policy: DecisionRevisionPolicy,
        state: DecisionRevisionState,
        verification_result: VerificationResult,
    ) -> DecisionRevisionDecision:
        """Evaluate revision eligibility for one verification result."""
        ...


def evaluate_decision_revision(
    *,
    policy: DecisionRevisionPolicy,
    state: DecisionRevisionState,
    verification_result: VerificationResult,
) -> DecisionRevisionDecision:
    """Canonical revision policy evaluation for one verification result."""
    if type(policy) is not DecisionRevisionPolicy:
        raise TypeError("policy must be DecisionRevisionPolicy")
    if type(state) is not DecisionRevisionState:
        raise TypeError("state must be DecisionRevisionState")
    if type(verification_result) is not VerificationResult:
        raise TypeError("verification_result must be VerificationResult")
    proposal_ref = verification_result.proposal_ref
    if verification_result.disposition is VerificationDisposition.PASSED:
        return DecisionRevisionDecision(
            disposition=DecisionRevisionDisposition.NOT_REQUIRED,
            proposal_ref=proposal_ref,
        )
    if verification_result.disposition is not VerificationDisposition.CHALLENGED:
        raise ValueError(
            "verification_result.disposition must be PASSED or CHALLENGED",
        )
    if state.revision_count < policy.max_revisions:
        return DecisionRevisionDecision(
            disposition=DecisionRevisionDisposition.ALLOWED,
            proposal_ref=proposal_ref,
            revision_number=state.revision_count + 1,
        )
    return DecisionRevisionDecision(
        disposition=DecisionRevisionDisposition.EXHAUSTED,
        proposal_ref=proposal_ref,
    )


def decision_revision_authorization(
    *,
    revision_decision: DecisionRevisionDecision,
    policy: DecisionRevisionPolicy,
) -> DecisionRevisionAuthorization:
    """Mint one revision authorization from an ALLOWED revision decision."""
    if type(revision_decision) is not DecisionRevisionDecision:
        raise TypeError("revision_decision must be DecisionRevisionDecision")
    if type(policy) is not DecisionRevisionPolicy:
        raise TypeError("policy must be DecisionRevisionPolicy")
    if revision_decision.disposition is not DecisionRevisionDisposition.ALLOWED:
        raise ValueError(
            "revision authorization requires DecisionRevisionDisposition.ALLOWED",
        )
    if revision_decision.revision_number is None:
        raise ValueError("revision authorization requires revision_number")
    return DecisionRevisionAuthorization(
        proposal_ref=revision_decision.proposal_ref,
        policy=policy,
        revision_number=revision_decision.revision_number,
    )
