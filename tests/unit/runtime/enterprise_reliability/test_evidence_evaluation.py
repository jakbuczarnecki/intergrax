# © Artur Czarnecki. All rights reserved.

"""ERL — evidence evaluation runtime integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from intergrax.contracts.enterprise_reliability import (
    EvidenceEvaluationOutcome,
    EvidenceEvaluationRequest,
    EvidenceEvaluatorAdvice,
    EvidenceEvaluatorStrategy,
    ExternalEffectEvidenceVerdict,
    ReconciliationProbeRequest,
    ReconciliationProbeResult,
    ResolutionDisposition,
    UnknownUncertaintyPosture,
)
from intergrax.runtime.enterprise_reliability import (
    admit_external_effect_unknown,
    evaluate_external_effect_evidence,
    materialize_external_effect_evidence_from_probe,
    plan_external_effect_resolution,
    EnterpriseReliabilityPluginGatewayImpl,
    InMemoryEnterpriseReliabilityPluginRegistry,
)
from tests.unit.runtime.enterprise_reliability.test_resolution_orchestration import (
    _contract,
)

pytestmark = pytest.mark.unit

_FIXED_TIME = datetime(2026, 9, 12, 8, 30, 0, tzinfo=UTC)


def _materialized_evidence(
    verdict: ExternalEffectEvidenceVerdict = ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS,
) -> object:
    return materialize_external_effect_evidence_from_probe(
        probe_request=ReconciliationProbeRequest(
            tenant_id="tenant-a",
            correlation_id="corr-pay",
            contract_id="pay-1",
            probe_ref="payment_status",
            plugin_id="reconcile-pay",
            attempt_index=1,
        ),
        probe_result=ReconciliationProbeResult(
            verdict=verdict,
            evidence_ref="evidence://pay/corr-pay/1",
        ),
        obtained_at=_FIXED_TIME,
    )


def test_reconciliation_output_enters_evaluation() -> None:
    state = admit_external_effect_unknown(correlation_id="corr-pay")
    evidence = _materialized_evidence()
    result = evaluate_external_effect_evidence(
        state=state,
        evidence=evidence,
        tenant_id="tenant-a",
        contract_id="pay-1",
    )

    assert result.outcome is EvidenceEvaluationOutcome.READY_FOR_DECISION


@dataclass(frozen=True, slots=True)
class _BrokenEvaluator:
    def evaluate(
        self,
        request: EvidenceEvaluationRequest,
    ) -> EvidenceEvaluatorAdvice | None:
        raise RuntimeError("evaluator_down")


def test_unavailable_evaluator_fails_closed() -> None:
    state = admit_external_effect_unknown(correlation_id="corr-pay")
    evidence = _materialized_evidence()
    result = evaluate_external_effect_evidence(
        state=state,
        evidence=evidence,
        tenant_id="tenant-a",
        contract_id="pay-1",
        evaluator_strategy=_BrokenEvaluator(),
    )

    assert result.outcome is EvidenceEvaluationOutcome.EVALUATION_FAILED
    assert result.rationale == "evaluator_unavailable"


def test_resolution_planning_carries_evaluation_result() -> None:
    state = admit_external_effect_unknown(correlation_id="corr-pay")
    gateway = EnterpriseReliabilityPluginGatewayImpl(
        InMemoryEnterpriseReliabilityPluginRegistry(),
    )
    planning = plan_external_effect_resolution(
        state=state,
        contract_id="pay-1",
        effect_contract=_contract(),
        unknown_posture=UnknownUncertaintyPosture.RECONCILE_ONLY,
        evidence=_materialized_evidence(
            verdict=ExternalEffectEvidenceVerdict.INSUFFICIENT,
        ),
        gateway=gateway,
        plugin_id="resolve-pay",
        tenant_id="tenant-a",
    )

    assert planning.evidence_evaluation.outcome is EvidenceEvaluationOutcome.INSUFFICIENT_EVIDENCE
    assert planning.plan.disposition is ResolutionDisposition.DEFER_INSUFFICIENT_EVIDENCE


def test_evidence_evaluator_strategy_protocol() -> None:
    @dataclass(frozen=True, slots=True)
    class _Stub(EvidenceEvaluatorStrategy):
        def evaluate(
            self,
            request: EvidenceEvaluationRequest,
        ) -> EvidenceEvaluatorAdvice | None:
            return None

    assert isinstance(_Stub(), EvidenceEvaluatorStrategy)
