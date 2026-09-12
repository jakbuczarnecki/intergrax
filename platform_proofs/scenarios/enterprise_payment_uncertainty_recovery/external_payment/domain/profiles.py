"""Data-driven execution profiles assembled from the logical dataset."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.domain.failures import (
    CommunicationFailureKind,
    resolve_communication_failure_kind,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.domain.sor_truth import (
    SorTruthFields,
)


@dataclass(frozen=True, slots=True)
class CommunicationUncertaintyProfile:
    lost_response: bool
    delayed_confirmation: bool
    uncertainty_cause: str
    failure_kind: CommunicationFailureKind
    narrative: str


@dataclass(frozen=True, slots=True)
class VariantExecutionProfile:
    variant_id: str
    effect_logical_id: str
    business_operation: str
    immediate_integration_outcome: str
    sor_truth: SorTruthFields
    communication: CommunicationUncertaintyProfile


def build_execution_profile(
    *,
    variant_id: str,
    variant_document: dict[str, object],
    external_effect: dict[str, object],
    communication_event: dict[str, object],
    sor_truth: SorTruthFields,
) -> VariantExecutionProfile:
    uncertainty_cause = str(communication_event.get("uncertainty_cause", ""))
    return VariantExecutionProfile(
        variant_id=variant_id,
        effect_logical_id=str(external_effect.get("effect_id", "")),
        business_operation=str(external_effect.get("business_operation", "capture_funds_for_order")),
        immediate_integration_outcome=str(
            external_effect.get("immediate_integration_outcome", "unknown")
        ),
        sor_truth=sor_truth,
        communication=CommunicationUncertaintyProfile(
            lost_response=bool(communication_event.get("lost_response")),
            delayed_confirmation=bool(communication_event.get("delayed_confirmation")),
            uncertainty_cause=uncertainty_cause,
            failure_kind=resolve_communication_failure_kind(uncertainty_cause),
            narrative=str(communication_event.get("narrative", "")),
        ),
    )
