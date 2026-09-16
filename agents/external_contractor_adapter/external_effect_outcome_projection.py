# © Artur Czarnecki. All rights reserved.

"""Project External Work provider observation → canonical ``ExternalEffectOutcome`` (GR-7-A2)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from external_contractor_adapter.schemas.adapt_result import ExternalWorkAdapterResult
from intergrax.contracts.enterprise_reliability.outcome import ExternalEffectOutcome
from intergrax.contracts.external_work import is_uncertain_external_work_outcome
from intergrax.contracts.runtime_policy import PolicyAction


class ExternalWorkSideEffectObservation(BaseModel):
    """Typed provider observation after meaningful side-effect governance — not authorization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_mutation_attempted: bool = Field(
        description="True only when a mutating provider call was dispatched after ALLOW.",
    )
    adapter_result: ExternalWorkAdapterResult
    reason: str = Field(default="", max_length=512)


def external_work_provider_mutation_attempted(
    adapter_result: ExternalWorkAdapterResult,
    *,
    policy_denied: bool,
) -> bool:
    """Return whether a mutating provider call occurred (Governance DENY → False)."""
    if policy_denied:
        return False
    decision = adapter_result.policy_decision
    if decision is not None and decision.action is not PolicyAction.ALLOW:
        return False
    if adapter_result.used and adapter_result.proof is not None:
        return True
    return adapter_result.error_code is not None


def project_external_work_side_effect_to_effect_outcome(
    observation: ExternalWorkSideEffectObservation,
) -> ExternalEffectOutcome | None:
    """
    Classify external-effect truth from adapter observation.

    Returns None when no mutating provider attempt occurred (no ERL admission).
    """
    if not observation.provider_mutation_attempted:
        return None
    result = observation.adapter_result
    decision = result.policy_decision
    if (
        result.used
        and result.proof is not None
        and decision is not None
        and decision.action is PolicyAction.ALLOW
    ):
        return ExternalEffectOutcome.SUCCESS
    code = result.error_code
    if code is not None and is_uncertain_external_work_outcome(code):
        return ExternalEffectOutcome.UNKNOWN
    if code is not None:
        return ExternalEffectOutcome.FAILURE
    return ExternalEffectOutcome.FAILURE


__all__ = [
    "ExternalWorkSideEffectObservation",
    "external_work_provider_mutation_attempted",
    "project_external_work_side_effect_to_effect_outcome",
]
