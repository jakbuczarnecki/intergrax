# © Artur Czarnecki. All rights reserved.

"""ERL — reconciliation evidence foundation tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.contracts.enterprise_reliability import (
    ExternalEffectEvidenceCheckResult,
    ExternalEffectEvidenceConfidence,
    ExternalEffectEvidenceError,
    ExternalEffectEvidenceSourceKind,
    ExternalEffectEvidenceType,
    ExternalEffectEvidenceVerdict,
    ExternalEffectOutcome,
    ReconciliationProbeRequest,
    ReconciliationProbeResult,
    UncertaintyLifecyclePhase,
    UncertaintyResolutionKind,
)
from intergrax.runtime.enterprise_reliability import (
    admit_external_effect_unknown,
    apply_reconciliation_evidence,
    materialize_external_effect_evidence_from_probe,
    resolve_uncertainty,
)

pytestmark = pytest.mark.unit

_FIXED_TIME = datetime(2026, 9, 12, 6, 0, 0, tzinfo=UTC)


def _probe_request() -> ReconciliationProbeRequest:
    return ReconciliationProbeRequest(
        tenant_id="tenant-a",
        correlation_id="corr-pay",
        contract_id="pay-1",
        probe_ref="payment_status",
        plugin_id="reconcile-pay",
        attempt_index=1,
    )


def test_materialize_evidence_from_definitive_probe_result() -> None:
    result = ReconciliationProbeResult(
        verdict=ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS,
        evidence_ref="evidence://pay/corr-pay/1",
        rationale="provider_read_ok",
    )
    evidence = materialize_external_effect_evidence_from_probe(
        probe_request=_probe_request(),
        probe_result=result,
        obtained_at=_FIXED_TIME,
    )

    assert evidence.source_kind is ExternalEffectEvidenceSourceKind.RECONCILIATION_PROBE
    assert evidence.evidence_type is ExternalEffectEvidenceType.RECONCILIATION_PROBE_READ
    assert evidence.confidence is ExternalEffectEvidenceConfidence.DEFINITIVE
    assert evidence.check_result is ExternalEffectEvidenceCheckResult.CONFIRMED_SUCCESS
    assert evidence.obtained_at == _FIXED_TIME
    assert evidence.operation_link.correlation_id == "corr-pay"
    assert evidence.operation_link.contract_id == "pay-1"
    assert evidence.evidence_ref == "evidence://pay/corr-pay/1"


def test_apply_evidence_preserves_operation_link_on_resolve() -> None:
    state = admit_external_effect_unknown(correlation_id="corr-pay")
    evidence = materialize_external_effect_evidence_from_probe(
        probe_request=_probe_request(),
        probe_result=ReconciliationProbeResult(
            verdict=ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS,
            evidence_ref="evidence://pay/corr-pay/1",
        ),
        obtained_at=_FIXED_TIME,
    )
    updated = apply_reconciliation_evidence(state, evidence)

    assert updated.effect_outcome is ExternalEffectOutcome.SUCCESS
    assert updated.lifecycle_phase is UncertaintyLifecyclePhase.RESOLVED
    assert updated.resolution_kind is UncertaintyResolutionKind.CONFIRMED_SUCCESS
    assert evidence.operation_link.correlation_id == updated.correlation_id


def test_inconclusive_evidence_keeps_unknown() -> None:
    state = admit_external_effect_unknown(correlation_id="corr-pay")
    evidence = materialize_external_effect_evidence_from_probe(
        probe_request=_probe_request(),
        probe_result=ReconciliationProbeResult(
            verdict=ExternalEffectEvidenceVerdict.INSUFFICIENT,
            evidence_ref="evidence://pay/corr-pay/insufficient",
        ),
    )
    updated = apply_reconciliation_evidence(state, evidence)

    assert updated.effect_outcome is ExternalEffectOutcome.UNKNOWN
    assert updated.lifecycle_phase is UncertaintyLifecyclePhase.ADMITTED


def test_mismatched_correlation_rejects_evidence() -> None:
    state = admit_external_effect_unknown(correlation_id="other-corr")
    evidence = materialize_external_effect_evidence_from_probe(
        probe_request=_probe_request(),
        probe_result=ReconciliationProbeResult(
            verdict=ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS,
            evidence_ref="evidence://pay/corr-pay/1",
        ),
    )
    with pytest.raises(ExternalEffectEvidenceError):
        apply_reconciliation_evidence(state, evidence)


def test_unknown_cannot_leave_without_definitive_evidence() -> None:
    state = admit_external_effect_unknown(correlation_id="corr-pay")
    with pytest.raises(ExternalEffectEvidenceError):
        apply_reconciliation_evidence(
            resolve_uncertainty(
                state,
                resolution_kind=UncertaintyResolutionKind.CONFIRMED_SUCCESS,
                resolved_outcome=ExternalEffectOutcome.SUCCESS,
            ),
            materialize_external_effect_evidence_from_probe(
                probe_request=_probe_request(),
                probe_result=ReconciliationProbeResult(
                    verdict=ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS,
                    evidence_ref="evidence://pay/corr-pay/1",
                ),
            ),
        )
