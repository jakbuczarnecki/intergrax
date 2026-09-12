# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Map reconciliation probe results to platform evidence and lifecycle handoff."""

from __future__ import annotations

from datetime import UTC, datetime

from intergrax.contracts.enterprise_reliability.evidence import (
    ExternalEffectEvidenceVerdict,
    classify_external_effect_outcome,
)
from intergrax.contracts.enterprise_reliability.lifecycle import UncertaintyStateRecord
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
from intergrax.runtime.enterprise_reliability.uncertainty_lifecycle import (
    resolve_uncertainty,
)


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


def assert_reconciliation_evidence_applicable(
    state: UncertaintyStateRecord,
    evidence: ExternalEffectEvidence,
) -> None:
    """
    Validate reconciliation evidence against an UNKNOWN episode.

    Does not close uncertainty — resolution strategies apply in a later orchestration step.
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


def apply_reconciliation_evidence(
    state: UncertaintyStateRecord,
    evidence: ExternalEffectEvidence,
) -> UncertaintyStateRecord:
    """
    Apply platform-default resolution for definitive reconciliation evidence.

    Prefer ``plan_external_effect_resolution`` + ``execute_external_effect_resolution`` for
    plugin-driven closure; this helper preserves the evidence-only fast path for tests.
    """
    assert_reconciliation_evidence_applicable(state, evidence)
    if evidence.verdict is ExternalEffectEvidenceVerdict.INSUFFICIENT:
        return state
    if evidence.confidence is not ExternalEffectEvidenceConfidence.DEFINITIVE:
        raise ExternalEffectEvidenceError(
            "definitive outcome change requires definitive evidence confidence",
        )
    from intergrax.contracts.enterprise_reliability.resolution import (
        resolution_advice_from_decision,
    )
    from intergrax.contracts.enterprise_reliability.resolution_decision import (
        ResolutionDecision,
        ResolutionPlatformAction,
    )

    advice = resolution_advice_from_decision(
        ResolutionDecision(
            action=ResolutionPlatformAction.CONTINUE,
            rationale="evidence_only_fast_path",
        ),
        evidence,
    )
    if advice is None:
        return state
    resolved_outcome = classify_external_effect_outcome(evidence.verdict)
    return resolve_uncertainty(
        state,
        resolution_kind=advice.resolution_kind,
        resolved_outcome=resolved_outcome,
    )


__all__ = [
    "apply_reconciliation_evidence",
    "assert_reconciliation_evidence_applicable",
    "materialize_external_effect_evidence_from_probe",
]
