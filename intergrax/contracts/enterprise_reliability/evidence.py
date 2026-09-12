# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Evidence sufficiency for external-effect outcome admission (ERL Phase 1)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from intergrax.contracts.enterprise_reliability.outcome import ExternalEffectOutcome
from intergrax.contracts.external_operation_cancellation import ExternalOperationPhysicalState

SCHEMA_EXTERNAL_EFFECT_EVIDENCE_VERDICT_V1: Final = "external_effect_evidence_verdict.v1"


class ExternalEffectEvidenceVerdict(StrEnum):
    """Whether observed evidence supports a definitive external-effect outcome."""

    DEFINITIVE_SUCCESS = "definitive_success"
    DEFINITIVE_FAILURE = "definitive_failure"
    INSUFFICIENT = "insufficient"


def classify_external_effect_outcome(
    verdict: ExternalEffectEvidenceVerdict,
) -> ExternalEffectOutcome:
    """Map evidence sufficiency to the canonical tri-state — never infer SUCCESS/FAILURE."""
    if verdict is ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS:
        return ExternalEffectOutcome.SUCCESS
    if verdict is ExternalEffectEvidenceVerdict.DEFINITIVE_FAILURE:
        return ExternalEffectOutcome.FAILURE
    return ExternalEffectOutcome.UNKNOWN


def external_effect_outcome_from_physical_state(
    physical_state: ExternalOperationPhysicalState,
) -> ExternalEffectOutcome | None:
    """Project durable external-operation physical state when terminal for effect truth."""
    if physical_state is ExternalOperationPhysicalState.SUCCEEDED:
        return ExternalEffectOutcome.SUCCESS
    if physical_state is ExternalOperationPhysicalState.FAILED:
        return ExternalEffectOutcome.FAILURE
    if physical_state is ExternalOperationPhysicalState.UNKNOWN:
        return ExternalEffectOutcome.UNKNOWN
    return None


__all__ = [
    "ExternalEffectEvidenceVerdict",
    "SCHEMA_EXTERNAL_EFFECT_EVIDENCE_VERDICT_V1",
    "classify_external_effect_outcome",
    "external_effect_outcome_from_physical_state",
]
