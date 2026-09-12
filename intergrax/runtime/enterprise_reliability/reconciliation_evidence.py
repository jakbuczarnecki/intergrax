# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Map reconciliation probe results to platform evidence and lifecycle handoff."""

from __future__ import annotations

from datetime import UTC, datetime

from intergrax.contracts.enterprise_reliability.evidence import (
    ExternalEffectEvidenceVerdict,
    classify_external_effect_outcome,
)
from intergrax.contracts.enterprise_reliability.lifecycle import (
    UncertaintyResolutionKind,
    UncertaintyStateRecord,
)
from intergrax.contracts.enterprise_reliability.outcome import ExternalEffectOutcome
from intergrax.contracts.enterprise_reliability.reconciliation_evidence import (
    ExternalEffectEvidence,
    ExternalEffectEvidenceConfidence,
    ExternalEffectEvidenceError,
    ExternalEffectEvidenceOperationLink,
    ExternalEffectEvidenceSourceKind,
    ExternalEffectEvidenceType,
    check_result_from_verdict,
    confidence_from_verdict,
)
from intergrax.contracts.enterprise_reliability.reconciliation_execution import (
    ReconciliationProbeRequest,
    ReconciliationProbeResult,
)
from intergrax.runtime.enterprise_reliability.uncertainty_lifecycle import resolve_uncertainty


def materialize_external_effect_evidence_from_probe(
    *,
    probe_request: ReconciliationProbeRequest,
    probe_result: ReconciliationProbeResult,
    obtained_at: datetime | None = None,
) -> ExternalEffectEvidence:
    """Translate plugin probe output into a platform evidence record."""
    timestamp = obtained_at or datetime.now(tz=UTC)
    return ExternalEffectEvidence(
        source_kind=ExternalEffectEvidenceSourceKind.RECONCILIATION_PROBE,
        evidence_type=ExternalEffectEvidenceType.RECONCILIATION_PROBE_READ,
        confidence=confidence_from_verdict(probe_result.verdict),
        check_result=check_result_from_verdict(probe_result.verdict),
        verdict=probe_result.verdict,
        evidence_ref=probe_result.evidence_ref,
        operation_link=ExternalEffectEvidenceOperationLink(
            tenant_id=probe_request.tenant_id,
            correlation_id=probe_request.correlation_id,
            contract_id=probe_request.contract_id,
            probe_ref=probe_request.probe_ref,
            plugin_id=probe_request.plugin_id,
            attempt_index=probe_request.attempt_index,
        ),
        obtained_at=timestamp,
        rationale=probe_result.rationale,
    )


def _resolution_kind_for_evidence(
    evidence: ExternalEffectEvidence,
) -> UncertaintyResolutionKind | None:
    if evidence.verdict is ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS:
        return UncertaintyResolutionKind.CONFIRMED_SUCCESS
    if evidence.verdict is ExternalEffectEvidenceVerdict.DEFINITIVE_FAILURE:
        return UncertaintyResolutionKind.CONFIRMED_FAILURE
    return None


def apply_reconciliation_evidence(
    state: UncertaintyStateRecord,
    evidence: ExternalEffectEvidence,
) -> UncertaintyStateRecord:
    """
    Advance uncertainty lifecycle using recorded reconciliation evidence.

    UNKNOWN may leave the episode only when evidence is definitive, operation-linked,
    and consistent with the current correlation identity.
    """
    if state.effect_outcome is not ExternalEffectOutcome.UNKNOWN:
        raise ExternalEffectEvidenceError(
            "reconciliation evidence applies only to UNKNOWN effect outcomes",
        )
    if evidence.operation_link.correlation_id != state.correlation_id:
        raise ExternalEffectEvidenceError(
            "evidence correlation_id does not match uncertainty episode",
        )
    if evidence.source_kind is not ExternalEffectEvidenceSourceKind.RECONCILIATION_PROBE:
        raise ExternalEffectEvidenceError("unsupported evidence source for reconciliation")

    resolution_kind = _resolution_kind_for_evidence(evidence)
    if resolution_kind is None:
        return state

    if evidence.confidence is not ExternalEffectEvidenceConfidence.DEFINITIVE:
        raise ExternalEffectEvidenceError(
            "definitive outcome change requires definitive evidence confidence",
        )

    resolved_outcome = classify_external_effect_outcome(evidence.verdict)
    return resolve_uncertainty(
        state,
        resolution_kind=resolution_kind,
        resolved_outcome=resolved_outcome,
    )


__all__ = [
    "apply_reconciliation_evidence",
    "materialize_external_effect_evidence_from_probe",
]
