"""Map scenario SoR snapshots to platform ``ReconciliationProbeResult``."""

from __future__ import annotations

from intergrax.contracts.enterprise_reliability.evidence import ExternalEffectEvidenceVerdict
from intergrax.contracts.enterprise_reliability.reconciliation_execution import (
    ReconciliationProbeRequest,
    ReconciliationProbeResult,
)

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.external_reality_lookup import (
    ExternalRealitySnapshot,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.failures import (
    ExternalRealityInconsistentState,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.constants import (
    EXTERNAL_EFFECT_SOR_PROBE_REF,
)

_TERMINAL_SUCCESS = "PAYMENT_COMPLETED"
_TERMINAL_FAILURE = "PAYMENT_FAILED"
_TERMINAL_INDETERMINATE = "TRUTH_INDETERMINATE"

_TRUTH_AVAILABLE = "AVAILABLE"
_TRUTH_UNAVAILABLE = "UNAVAILABLE"
_TRUTH_INDETERMINATE = "INDETERMINATE"


def _evidence_ref(request: ReconciliationProbeRequest, snapshot: ExternalRealitySnapshot) -> str:
    suffix = snapshot.sor_transaction_ref or snapshot.external_effect_reference
    return (
        f"evidence://erl-qual-004/reconcile/"
        f"{request.correlation_id}/{request.attempt_index}/{suffix}"
    )


def _assert_consistent_terminal_state(snapshot: ExternalRealitySnapshot) -> None:
    outcome = snapshot.terminal_outcome
    captured = snapshot.funds_captured
    if outcome == _TERMINAL_SUCCESS and not captured:
        raise ExternalRealityInconsistentState(
            "terminal_outcome PAYMENT_COMPLETED requires funds_captured=true",
        )
    if outcome == _TERMINAL_FAILURE and captured:
        raise ExternalRealityInconsistentState(
            "terminal_outcome PAYMENT_FAILED requires funds_captured=false",
        )


def map_snapshot_to_probe_result(
    *,
    request: ReconciliationProbeRequest,
    snapshot: ExternalRealitySnapshot,
) -> ReconciliationProbeResult:
    """Translate external reality into generic reconciliation evidence."""
    if request.probe_ref != EXTERNAL_EFFECT_SOR_PROBE_REF:
        return ReconciliationProbeResult(
            verdict=ExternalEffectEvidenceVerdict.INSUFFICIENT,
            evidence_ref=_evidence_ref(request, snapshot),
            rationale=f"unsupported_probe_ref:{request.probe_ref}",
        )

    if snapshot.correlation_id != request.correlation_id:
        return ReconciliationProbeResult(
            verdict=ExternalEffectEvidenceVerdict.INSUFFICIENT,
            evidence_ref=_evidence_ref(request, snapshot),
            rationale="correlation_id_mismatch",
        )

    availability = snapshot.truth_availability_state
    if availability in {_TRUTH_UNAVAILABLE, _TRUTH_INDETERMINATE}:
        return ReconciliationProbeResult(
            verdict=ExternalEffectEvidenceVerdict.INSUFFICIENT,
            evidence_ref=_evidence_ref(request, snapshot),
            rationale=f"truth_availability_state:{availability}",
        )

    if availability != _TRUTH_AVAILABLE:
        return ReconciliationProbeResult(
            verdict=ExternalEffectEvidenceVerdict.INSUFFICIENT,
            evidence_ref=_evidence_ref(request, snapshot),
            rationale=f"unsupported_truth_availability_state:{availability}",
        )

    try:
        _assert_consistent_terminal_state(snapshot)
    except ExternalRealityInconsistentState as exc:
        return ReconciliationProbeResult(
            verdict=ExternalEffectEvidenceVerdict.INSUFFICIENT,
            evidence_ref=_evidence_ref(request, snapshot),
            rationale=str(exc),
        )

    outcome = snapshot.terminal_outcome
    if outcome == _TERMINAL_SUCCESS:
        return ReconciliationProbeResult(
            verdict=ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS,
            evidence_ref=_evidence_ref(request, snapshot),
            rationale="sor_terminal_outcome_confirms_success",
        )
    if outcome == _TERMINAL_FAILURE:
        return ReconciliationProbeResult(
            verdict=ExternalEffectEvidenceVerdict.DEFINITIVE_FAILURE,
            evidence_ref=_evidence_ref(request, snapshot),
            rationale="sor_terminal_outcome_confirms_failure",
        )
    if outcome == _TERMINAL_INDETERMINATE:
        return ReconciliationProbeResult(
            verdict=ExternalEffectEvidenceVerdict.INSUFFICIENT,
            evidence_ref=_evidence_ref(request, snapshot),
            rationale="sor_terminal_outcome_indeterminate",
        )

    return ReconciliationProbeResult(
        verdict=ExternalEffectEvidenceVerdict.INSUFFICIENT,
        evidence_ref=_evidence_ref(request, snapshot),
        rationale=f"unsupported_terminal_outcome:{outcome}",
    )
