# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Boundary between ERL outcome truth and Reliability failure classification."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.enterprise_reliability.effect_contract import (
    ExternalEffectContract,
    contract_declares_idempotency,
)
from intergrax.contracts.enterprise_reliability.outcome import ExternalEffectOutcome
from intergrax.contracts.execution_retry import (
    ExecutionFailureClassification,
    ExecutionFailureKind,
)
from intergrax.contracts.resilience_policy import FailureClass

_MAX_REASON = 512


class ExternalEffectReliabilityInteraction(StrEnum):
    """Whether Reliability retry/failure projection applies."""

    NO_FAILURE_CLASSIFICATION = "no_failure_classification"
    DEFINITIVE_FAILURE = "definitive_failure"
    UNCERTAINTY_FAIL_CLOSED = "uncertainty_fail_closed"


class ExternalEffectReliabilityProjection(BaseModel):
    """Typed bridge from external-effect outcome to Reliability inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    interaction: ExternalEffectReliabilityInteraction
    classification: ExecutionFailureClassification | None = None
    reason: str = Field(default="", max_length=_MAX_REASON)


def project_external_effect_to_reliability(
    outcome: ExternalEffectOutcome,
    *,
    side_effect_idempotency_guaranteed: bool = False,
    effect_contract: ExternalEffectContract | None = None,
    reason: str = "",
) -> ExternalEffectReliabilityProjection:
    """
    Map canonical outcome to Reliability without collapsing UNKNOWN to DEPENDENCY_ERROR.

    UNKNOWN stays ``ExecutionFailureKind.UNKNOWN`` with optional unknown-side-effect flag —
    not a retryable transient failure.

    When ``effect_contract`` is supplied, idempotency for UNKNOWN projection follows the
    contract declaration; otherwise ``side_effect_idempotency_guaranteed`` applies.
    """
    idempotency_guaranteed = (
        contract_declares_idempotency(effect_contract)
        if effect_contract is not None
        else side_effect_idempotency_guaranteed
    )
    bounded_reason = reason[:_MAX_REASON]
    if outcome is ExternalEffectOutcome.SUCCESS:
        return ExternalEffectReliabilityProjection(
            interaction=ExternalEffectReliabilityInteraction.NO_FAILURE_CLASSIFICATION,
            reason=bounded_reason or "external_effect_success",
        )
    if outcome is ExternalEffectOutcome.FAILURE:
        return ExternalEffectReliabilityProjection(
            interaction=ExternalEffectReliabilityInteraction.DEFINITIVE_FAILURE,
            classification=ExecutionFailureClassification(
                kind=ExecutionFailureKind.NON_RETRYABLE_PERMANENT,
                reason=bounded_reason or "external_effect_failure",
                failure_class=FailureClass.DEPENDENCY_ERROR,
            ),
            reason=bounded_reason or "external_effect_failure",
        )
    return ExternalEffectReliabilityProjection(
        interaction=ExternalEffectReliabilityInteraction.UNCERTAINTY_FAIL_CLOSED,
        classification=ExecutionFailureClassification(
            kind=ExecutionFailureKind.UNKNOWN,
            reason=bounded_reason or "external_effect_unknown",
            has_unknown_side_effect=not idempotency_guaranteed,
        ),
        reason=bounded_reason or "external_effect_unknown",
    )


__all__ = [
    "ExternalEffectReliabilityInteraction",
    "ExternalEffectReliabilityProjection",
    "project_external_effect_to_reliability",
]
