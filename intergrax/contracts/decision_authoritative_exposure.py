# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Public authoritative Decision exposure contracts (P0-B-D1).

The types in this module represent a platform-issued authoritative outcome when
received from the trusted execution boundary. Construction of an equivalent
value by application code does not constitute platform-issued authority.

Type validity (structural invariants) is separate from authority authenticity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

from intergrax.contracts.decision_record import AuthoritativeAcceptedDecision
from intergrax.contracts.decision_resolution import AuthoritativeResolutionRecord

T = TypeVar("T")


class DecisionEvaluationScope(str, Enum):
    """Contracts-layer host evaluation scope (not runtime DecisionFlowScope)."""

    GRAPH_FINAL = "graph_final"
    UAEP_STEP = "uaep_step"


class ExposureUnevaluatedReason(str, Enum):
    """Terminal execution without a publishable authoritative Decision outcome."""

    NO_DECISION_GATE = "no_decision_gate"
    SCOPE_NOT_EVALUATED = "scope_not_evaluated"
    EXECUTION_FAILED_BEFORE_DECISION = "execution_failed_before_decision"
    EXECUTION_CANCELLED_BEFORE_DECISION = "execution_cancelled_before_decision"


@dataclass(frozen=True, slots=True)
class ExposureAccepted(Generic[T]):
    """Authoritative accepted decision for one evaluation scope."""

    scope: DecisionEvaluationScope
    accepted: AuthoritativeAcceptedDecision[T]

    def __post_init__(self) -> None:
        if type(self.scope) is not DecisionEvaluationScope:
            raise TypeError("ExposureAccepted.scope must be DecisionEvaluationScope")
        if type(self.accepted) is not AuthoritativeAcceptedDecision:
            raise TypeError(
                "ExposureAccepted.accepted must be AuthoritativeAcceptedDecision",
            )


@dataclass(frozen=True, slots=True)
class ExposureResolution:
    """Authoritative resolution when no decision version was accepted."""

    scope: DecisionEvaluationScope
    resolution: AuthoritativeResolutionRecord

    def __post_init__(self) -> None:
        if type(self.scope) is not DecisionEvaluationScope:
            raise TypeError("ExposureResolution.scope must be DecisionEvaluationScope")
        if type(self.resolution) is not AuthoritativeResolutionRecord:
            raise TypeError(
                "ExposureResolution.resolution must be AuthoritativeResolutionRecord",
            )


@dataclass(frozen=True, slots=True)
class ExposureUnevaluated:
    """Terminal task without publishable authoritative Decision for the host scope."""

    scope: DecisionEvaluationScope | None
    reason: ExposureUnevaluatedReason

    def __post_init__(self) -> None:
        if self.scope is not None and type(self.scope) is not DecisionEvaluationScope:
            raise TypeError(
                "ExposureUnevaluated.scope must be DecisionEvaluationScope or None",
            )
        if type(self.reason) is not ExposureUnevaluatedReason:
            raise TypeError(
                "ExposureUnevaluated.reason must be ExposureUnevaluatedReason",
            )


AuthoritativeDecisionExposure = ExposureAccepted[T] | ExposureResolution | ExposureUnevaluated


def validate_decision_evaluation_scope(value: object) -> DecisionEvaluationScope:
    if type(value) is DecisionEvaluationScope:
        return value
    if type(value) is not str:
        raise TypeError(
            f"DecisionEvaluationScope must be str or enum, got {type(value).__name__}",
        )
    return DecisionEvaluationScope(value)


def validate_exposure_unevaluated_reason(value: object) -> ExposureUnevaluatedReason:
    if type(value) is ExposureUnevaluatedReason:
        return value
    if type(value) is not str:
        raise TypeError(
            f"ExposureUnevaluatedReason must be str or enum, got {type(value).__name__}",
        )
    return ExposureUnevaluatedReason(value)
