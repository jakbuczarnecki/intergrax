# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ERL admission boundary — external effect entry and reliability case initialization."""

from __future__ import annotations

from intergrax.contracts.enterprise_reliability.admission_boundary import (
    ExternalEffectAdmissionCaseError,
    ExternalEffectAdmissionContextError,
    ExternalEffectAdmissionPhase,
    ExternalEffectAdmissionRequest,
    ExternalEffectAdmissionResult,
    assert_external_effect_admission_request,
    reliability_case_id_for_admission,
    uncertainty_state_ref_for_correlation,
)
from intergrax.contracts.enterprise_reliability.case_lifecycle import (
    ReliabilityCaseLifecycleContextError,
    ReliabilityCaseLifecycleState,
    ReliabilityCaseLifecycleTransitionRequest,
    initial_reliability_case_lifecycle,
)
from intergrax.contracts.enterprise_reliability.effect_contract import (
    validate_external_effect_contract,
)
from intergrax.contracts.enterprise_reliability.outcome import ExternalEffectOutcome
from intergrax.contracts.enterprise_reliability.reliability_boundary import (
    ExternalEffectReliabilityInteraction,
    project_external_effect_to_reliability,
)
from intergrax.runtime.enterprise_reliability.case_lifecycle_coordination import (
    transition_reliability_case_lifecycle,
)
from intergrax.runtime.enterprise_reliability.contract_admission import (
    admit_external_effect_unknown_with_contract,
)


def admit_external_effect_into_enterprise_reliability(
    request: ExternalEffectAdmissionRequest,
) -> ExternalEffectAdmissionResult:
    """
    Admit a domain-neutral external effect into ERL.

    Owns entry, classification, and reliability case initialization for UNKNOWN
    episodes. Does not run reconciliation, evidence, resolution, or recovery.
    """
    assert_external_effect_admission_request(request)
    validate_external_effect_contract(request.contract)

    projection = project_external_effect_to_reliability(
        request.effect_outcome,
        effect_contract=request.contract,
        reason=request.reason,
    )

    if (
        projection.interaction
        is not ExternalEffectReliabilityInteraction.UNCERTAINTY_FAIL_CLOSED
    ):
        return ExternalEffectAdmissionResult(
            phase=ExternalEffectAdmissionPhase.HANDED_OFF,
            external_effect_ref=request.external_effect_ref,
            correlation_id=request.correlation_id,
            contract_id=request.contract.contract_id,
            projection=projection,
        )

    if request.effect_outcome is not ExternalEffectOutcome.UNKNOWN:
        raise ExternalEffectAdmissionContextError(
            "uncertainty projection requires UNKNOWN effect_outcome",
        )

    unknown_admission = admit_external_effect_unknown_with_contract(
        correlation_id=request.correlation_id,
        contract=request.contract,
    )
    if unknown_admission.state.correlation_id != request.correlation_id:
        raise ExternalEffectAdmissionCaseError("uncertainty admission correlation mismatch")

    uncertainty_ref = (
        request.uncertainty_state_ref
        or uncertainty_state_ref_for_correlation(request.correlation_id)
    )
    case_id = request.case_id or reliability_case_id_for_admission(
        correlation_id=request.correlation_id,
        contract_id=request.contract.contract_id,
    )
    try:
        case_record = initial_reliability_case_lifecycle(
            case_id=case_id,
            correlation_id=request.correlation_id,
            contract_id=request.contract.contract_id,
            uncertainty_state_ref=uncertainty_ref,
        )
    except ReliabilityCaseLifecycleContextError as exc:
        raise ExternalEffectAdmissionCaseError(
            f"reliability case initialization failed: {exc}",
        ) from exc

    handoff = transition_reliability_case_lifecycle(
        ReliabilityCaseLifecycleTransitionRequest(
            record=case_record,
            target_state=ReliabilityCaseLifecycleState.RECONCILIATION_RUNNING,
            refs=case_record.refs,
        ),
    )

    return ExternalEffectAdmissionResult(
        phase=ExternalEffectAdmissionPhase.HANDED_OFF,
        external_effect_ref=request.external_effect_ref,
        correlation_id=request.correlation_id,
        contract_id=request.contract.contract_id,
        projection=projection,
        uncertainty_state=unknown_admission.state,
        unknown_posture=unknown_admission.unknown_posture,
        case_record=handoff.record,
    )


__all__ = [
    "ExternalEffectAdmissionContextError",
    "ExternalEffectAdmissionCaseError",
    "admit_external_effect_into_enterprise_reliability",
]
