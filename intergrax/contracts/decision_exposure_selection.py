# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution-host publication and selection contracts for Decision exposure (P0-B-D1-R1).

Selection operates only on candidates from one effective execution attempt.
Execution Engine resolves the effective attempt before invoking a strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, Protocol, TypeVar, runtime_checkable

from intergrax.contracts.decision_authoritative_exposure import (
    AuthoritativeDecisionExposure,
    DecisionEvaluationScope,
    validate_decision_evaluation_scope,
)
from intergrax.contracts.decision_identity import DecisionExecutionLineage, DecisionScope

T = TypeVar("T")

DECISION_EXPOSURE_SELECTION_STRATEGY_CAPABILITY_ID = "decision.exposure_selection_strategy"


class HostPublicationClass(str, Enum):
    """Host classification for one mapped exposure fragment."""

    INTERMEDIATE = "intermediate"
    HOST_TERMINAL_CANDIDATE = "host_terminal_candidate"
    NON_PUBLISHABLE = "non_publishable"


class DecisionExposureSelectionFailureCode(str, Enum):
    """Structured fail-closed selection outcomes."""

    NO_ELIGIBLE_TERMINAL_CANDIDATE = "no_eligible_terminal_candidate"
    AMBIGUOUS_TERMINAL_CANDIDATES = "ambiguous_terminal_candidates"
    INVALID_CANDIDATE_SET = "invalid_candidate_set"
    INVARIANT_VIOLATION = "invariant_violation"


class DecisionExposureSelectionSuccessReason(str, Enum):
    """Structured success reason for deterministic host terminal selection."""

    HOST_TERMINAL_SINGLE_ELIGIBLE_CANDIDATE = "host_terminal_single_eligible_candidate"


@dataclass(frozen=True, slots=True)
class DecisionExposurePublicationPolicy:
    """Immutable host-declared terminal scope eligibility."""

    eligible_terminal_scopes: frozenset[DecisionEvaluationScope]

    def __post_init__(self) -> None:
        if type(self.eligible_terminal_scopes) is not frozenset:
            raise TypeError(
                "DecisionExposurePublicationPolicy.eligible_terminal_scopes "
                "must be frozenset",
            )
        for item in self.eligible_terminal_scopes:
            validate_decision_evaluation_scope(item)


@dataclass(frozen=True, slots=True)
class DecisionExposureCandidate(Generic[T]):
    """One mapped exposure fragment considered for host publication."""

    evaluation_scope: DecisionEvaluationScope
    decision_scope: DecisionScope
    execution_lineage: DecisionExecutionLineage
    host_publication_class: HostPublicationClass
    exposure: AuthoritativeDecisionExposure[T]
    evaluation_ordinal: int

    def __post_init__(self) -> None:
        validate_decision_evaluation_scope(self.evaluation_scope)
        if type(self.decision_scope) is not DecisionScope:
            raise TypeError(
                "DecisionExposureCandidate.decision_scope must be DecisionScope",
            )
        if type(self.execution_lineage) is not DecisionExecutionLineage:
            raise TypeError(
                "DecisionExposureCandidate.execution_lineage must be "
                "DecisionExecutionLineage",
            )
        if type(self.host_publication_class) is not HostPublicationClass:
            raise TypeError(
                "DecisionExposureCandidate.host_publication_class must be "
                "HostPublicationClass",
            )
        if self.evaluation_ordinal < 0:
            raise ValueError("evaluation_ordinal must be >= 0")


@dataclass(frozen=True, slots=True)
class DecisionExposureCandidateAppend(Generic[T]):
    """Collector input without evaluation_ordinal (assigned by collector)."""

    evaluation_scope: DecisionEvaluationScope
    decision_scope: DecisionScope
    execution_lineage: DecisionExecutionLineage
    host_publication_class: HostPublicationClass
    exposure: AuthoritativeDecisionExposure[T]

    def __post_init__(self) -> None:
        validate_decision_evaluation_scope(self.evaluation_scope)
        if type(self.decision_scope) is not DecisionScope:
            raise TypeError(
                "DecisionExposureCandidateAppend.decision_scope must be DecisionScope",
            )
        if type(self.execution_lineage) is not DecisionExecutionLineage:
            raise TypeError(
                "DecisionExposureCandidateAppend.execution_lineage must be "
                "DecisionExecutionLineage",
            )
        if type(self.host_publication_class) is not HostPublicationClass:
            raise TypeError(
                "DecisionExposureCandidateAppend.host_publication_class must be "
                "HostPublicationClass",
            )


@dataclass(frozen=True, slots=True)
class DecisionExposureSelectionDecision(Generic[T]):
    """Typed host selection outcome."""

    selected: AuthoritativeDecisionExposure[T]
    reason_code: DecisionExposureSelectionSuccessReason
    considered_candidates: int

    def __post_init__(self) -> None:
        if self.considered_candidates < 0:
            raise ValueError("considered_candidates must be >= 0")
        if type(self.reason_code) is not DecisionExposureSelectionSuccessReason:
            raise TypeError(
                "reason_code must be DecisionExposureSelectionSuccessReason",
            )


@dataclass(frozen=True, slots=True)
class DecisionExposureSelectionFailure:
    """Fail-closed selection result (not an exception)."""

    reason_code: DecisionExposureSelectionFailureCode
    detail: str

    def __post_init__(self) -> None:
        if type(self.reason_code) is not DecisionExposureSelectionFailureCode:
            raise TypeError(
                "reason_code must be DecisionExposureSelectionFailureCode",
            )
        if not self.detail or not self.detail.strip():
            raise ValueError("detail must be non-empty")


@runtime_checkable
class DecisionExposureSelectionStrategy(Protocol[T]):
    """Public extension seam for host publication selection."""

    @property
    def strategy_id(self) -> str:
        """Stable strategy identity for composition and plugin admission."""
        ...

    def select(
        self,
        policy: DecisionExposurePublicationPolicy,
        candidates: tuple[DecisionExposureCandidate[T], ...],
    ) -> DecisionExposureSelectionDecision[T] | DecisionExposureSelectionFailure:
        """Select one public exposure from candidates of one effective attempt."""
        ...
