# © Artur Czarnecki. All rights reserved.

"""ERL evidence evaluation — contract tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.contracts.enterprise_reliability import (
    EvidenceEvaluationOutcome,
    ExternalEffectEvidenceVerdict,
    ExternalEffectOutcome,
    ReconciliationProbeRequest,
    ReconciliationProbeResult,
    build_evidence_evaluation_context,
    evaluate_evidence_collection,
    initial_uncertainty_state,
)
from intergrax.runtime.enterprise_reliability import (
    materialize_external_effect_evidence_from_probe,
)

pytestmark = pytest.mark.unit

_FIXED_TIME = datetime(2026, 9, 12, 8, 0, 0, tzinfo=UTC)


def _probe_request() -> ReconciliationProbeRequest:
    return ReconciliationProbeRequest(
        tenant_id="tenant-a",
        correlation_id="corr-1",
        contract_id="contract-1",
        probe_ref="external_read",
        plugin_id="reconcile-1",
        attempt_index=1,
    )


def _evidence(
    *,
    verdict: ExternalEffectEvidenceVerdict,
    correlation_id: str = "corr-1",
) -> object:
    request = _probe_request().model_copy(update={"correlation_id": correlation_id})
    return materialize_external_effect_evidence_from_probe(
        probe_request=request,
        probe_result=ReconciliationProbeResult(
            verdict=verdict,
            evidence_ref=f"evidence://{correlation_id}/1",
        ),
        obtained_at=_FIXED_TIME,
    )


def test_valid_evidence_is_ready_for_decision() -> None:
    state = initial_uncertainty_state(correlation_id="corr-1")
    evidence = _evidence(verdict=ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS)
    context = build_evidence_evaluation_context(
        tenant_id="tenant-a",
        correlation_id="corr-1",
        contract_id="contract-1",
        evidence_items=(evidence,),
    )
    result = evaluate_evidence_collection(state=state, context=context)

    assert result.outcome is EvidenceEvaluationOutcome.READY_FOR_DECISION
    assert result.primary_evidence_ref == "evidence://corr-1/1"


def test_missing_evidence_is_insufficient() -> None:
    state = initial_uncertainty_state(correlation_id="corr-1")
    context = build_evidence_evaluation_context(
        tenant_id="tenant-a",
        correlation_id="corr-1",
        contract_id="contract-1",
        evidence_items=(),
    )
    result = evaluate_evidence_collection(state=state, context=context)

    assert result.outcome is EvidenceEvaluationOutcome.INSUFFICIENT_EVIDENCE


def test_conflicting_definitive_verdicts() -> None:
    state = initial_uncertainty_state(correlation_id="corr-1")
    success = _evidence(verdict=ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS)
    failure = _evidence(verdict=ExternalEffectEvidenceVerdict.DEFINITIVE_FAILURE)
    context = build_evidence_evaluation_context(
        tenant_id="tenant-a",
        correlation_id="corr-1",
        contract_id="contract-1",
        evidence_items=(success, failure),
    )
    result = evaluate_evidence_collection(state=state, context=context)

    assert result.outcome is EvidenceEvaluationOutcome.CONFLICTING_EVIDENCE


def test_inconclusive_evidence_is_insufficient() -> None:
    state = initial_uncertainty_state(correlation_id="corr-1")
    evidence = _evidence(verdict=ExternalEffectEvidenceVerdict.INSUFFICIENT)
    context = build_evidence_evaluation_context(
        tenant_id="tenant-a",
        correlation_id="corr-1",
        contract_id="contract-1",
        evidence_items=(evidence,),
    )
    result = evaluate_evidence_collection(state=state, context=context)

    assert result.outcome is EvidenceEvaluationOutcome.INSUFFICIENT_EVIDENCE


def test_evaluation_fails_when_episode_not_unknown() -> None:
    from intergrax.contracts.enterprise_reliability import UncertaintyResolutionKind
    from intergrax.runtime.enterprise_reliability import resolve_uncertainty

    state = resolve_uncertainty(
        initial_uncertainty_state(correlation_id="corr-1"),
        resolution_kind=UncertaintyResolutionKind.CONFIRMED_SUCCESS,
        resolved_outcome=ExternalEffectOutcome.SUCCESS,
    )
    evidence = _evidence(verdict=ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS)
    context = build_evidence_evaluation_context(
        tenant_id="tenant-a",
        correlation_id="corr-1",
        contract_id="contract-1",
        evidence_items=(evidence,),
    )
    result = evaluate_evidence_collection(state=state, context=context)

    assert result.outcome is EvidenceEvaluationOutcome.EVALUATION_FAILED
