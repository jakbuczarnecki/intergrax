# © Artur Czarnecki. All rights reserved.

"""Payment evidence evaluator plugin — SPI, rules, and reconciliation integration."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.contracts.enterprise_reliability import (
    EvidenceEvaluationOutcome,
    EvidenceEvaluatorStrategy,
    ExternalEffectEvidenceVerdict,
)
from intergrax.runtime.enterprise_reliability import (
    admit_external_effect_unknown,
    evaluate_external_effect_evidence,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.adapters.in_memory_payment_evidence_lookup import (
    InMemoryPaymentReconciliationEvidenceLookup,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.constants import (
    SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.external_reality_lookup import (
    ExternalRealitySnapshot,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.payment_reconciliation_evidence import (
    PaymentReconciliationEvidence,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.mapping.payment_evidence_fields import (
    resolve_payment_reconciliation_evidence,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.plugins.payment_evidence_evaluator import (
    PaymentEvidenceEvaluatorPlugin,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.domain.sor_truth import (
    resolve_sor_truth_fields,
)
from tests.unit.platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.test_reconciliation_plugin_integration import (
    _run_reconciliation,
    _seed_lookup,
)

pytestmark = pytest.mark.unit

_SCENARIO_ROOT = (
    Path(__file__).resolve().parents[5]
    / "platform_proofs/scenarios/enterprise_payment_uncertainty_recovery"
)
_FIXED_TIME = datetime(2026, 9, 12, 12, 0, 0, tzinfo=UTC)


def _variant_document(variant_id: str) -> dict:
    path = _SCENARIO_ROOT / "dataset/variants" / variant_id / "scenario_variant.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_payment_evidence(
    variant_id: str,
    correlation_id: str,
    *,
    snapshot: ExternalRealitySnapshot,
) -> PaymentReconciliationEvidence:
    document = _variant_document(variant_id)
    return resolve_payment_reconciliation_evidence(
        document,
        correlation_id=correlation_id,
        external_effect_reference=snapshot.external_effect_reference,
        sor_transaction_ref=snapshot.sor_transaction_ref,
        funds_captured=snapshot.funds_captured,
        truth_availability_state=snapshot.truth_availability_state,
        observed_at=_FIXED_TIME,
    )


def _evaluator_for_variant(variant_id: str, correlation_id: str) -> PaymentEvidenceEvaluatorPlugin:
    snapshot_fields = resolve_sor_truth_fields(_variant_document(variant_id))
    snapshot = ExternalRealitySnapshot(
        correlation_id=correlation_id,
        external_effect_reference="EXT-LAB",
        terminal_outcome=snapshot_fields.terminal_outcome,
        funds_captured=snapshot_fields.funds_captured,
        truth_availability_state=snapshot_fields.truth_availability_state,
        sor_transaction_ref="SOR-LAB",
    )
    payment_lookup = InMemoryPaymentReconciliationEvidenceLookup()
    payment_lookup.seed(_seed_payment_evidence(variant_id, correlation_id, snapshot=snapshot))
    return PaymentEvidenceEvaluatorPlugin(_lookup=payment_lookup)


def test_successful_payment_evidence_evaluation_ready_for_decision() -> None:
    correlation_id = "corr-pay-success"
    variant_id = "payment_completed_after_unknown"
    lookup = _seed_lookup(variant_id, correlation_id)
    _planning, run = _run_reconciliation(lookup, correlation_id=correlation_id)
    assert run.evidence is not None

    state = admit_external_effect_unknown(correlation_id=correlation_id)
    evaluator = _evaluator_for_variant(variant_id, correlation_id)
    result = evaluate_external_effect_evidence(
        state=state,
        evidence=run.evidence,
        tenant_id="tenant-lab",
        contract_id=SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
        evaluator_strategy=evaluator,
    )

    assert result.outcome is EvidenceEvaluationOutcome.READY_FOR_DECISION
    assert "payment_evidence_consistent" in result.rationale


def test_insufficient_payment_evidence_truth_unavailable() -> None:
    correlation_id = "corr-pay-insufficient"
    variant_id = "payment_truth_unavailable"
    lookup = _seed_lookup(variant_id, correlation_id)
    _planning, run = _run_reconciliation(lookup, correlation_id=correlation_id)
    assert run.evidence is not None

    state = admit_external_effect_unknown(correlation_id=correlation_id)
    evaluator = _evaluator_for_variant(variant_id, correlation_id)
    result = evaluate_external_effect_evidence(
        state=state,
        evidence=run.evidence,
        tenant_id="tenant-lab",
        contract_id=SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
        evaluator_strategy=evaluator,
    )

    assert result.outcome is EvidenceEvaluationOutcome.INSUFFICIENT_EVIDENCE


def test_conflicting_payment_evidence_probe_settlement_mismatch() -> None:
    correlation_id = "corr-pay-conflict"
    variant_id = "payment_completed_after_unknown"
    lookup = _seed_lookup(variant_id, correlation_id)
    _planning, run = _run_reconciliation(lookup, correlation_id=correlation_id)
    assert run.evidence is not None

    payment_lookup = InMemoryPaymentReconciliationEvidenceLookup()
    payment_lookup.seed(
        PaymentReconciliationEvidence(
            correlation_id=correlation_id,
            external_effect_reference="EXT-LAB",
            psp_confirmation_id="psp-conf-SOR-LAB",
            sor_transaction_ref="SOR-LAB",
            reconciliation_availability="available",
            discoverable_outcome="payment_succeeded",
            funds_captured=False,
            settlement_status="NOT_SETTLED",
            settlement_batch_ref=None,
            source_reliability_tier="AUTHORITATIVE",
            evidence_observed_at=_FIXED_TIME,
        ),
    )
    evaluator = PaymentEvidenceEvaluatorPlugin(_lookup=payment_lookup)

    state = admit_external_effect_unknown(correlation_id=correlation_id)
    result = evaluate_external_effect_evidence(
        state=state,
        evidence=run.evidence,
        tenant_id="tenant-lab",
        contract_id=SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
        evaluator_strategy=evaluator,
    )

    assert result.outcome is EvidenceEvaluationOutcome.CONFLICTING_EVIDENCE
    assert "payment_probe_settlement_mismatch" in result.rationale


def test_failed_payment_path_ready_for_negative_resolution() -> None:
    correlation_id = "corr-pay-failed"
    variant_id = "payment_failed_after_unknown"
    lookup = _seed_lookup(variant_id, correlation_id)
    _planning, run = _run_reconciliation(lookup, correlation_id=correlation_id)
    assert run.evidence is not None
    assert run.evidence.verdict is ExternalEffectEvidenceVerdict.DEFINITIVE_FAILURE

    state = admit_external_effect_unknown(correlation_id=correlation_id)
    evaluator = _evaluator_for_variant(variant_id, correlation_id)
    result = evaluate_external_effect_evidence(
        state=state,
        evidence=run.evidence,
        tenant_id="tenant-lab",
        contract_id=SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
        evaluator_strategy=evaluator,
    )

    assert result.outcome is EvidenceEvaluationOutcome.READY_FOR_DECISION


def test_plugin_implements_evidence_evaluator_strategy_protocol() -> None:
    lookup = InMemoryPaymentReconciliationEvidenceLookup()
    plugin = PaymentEvidenceEvaluatorPlugin(_lookup=lookup)
    assert isinstance(plugin, EvidenceEvaluatorStrategy)


def test_evaluator_abstains_for_non_scenario_evidence_refs() -> None:
    correlation_id = "corr-other"
    lookup = _seed_lookup("payment_completed_after_unknown", correlation_id)
    _planning, run = _run_reconciliation(lookup, correlation_id=correlation_id)
    assert run.evidence is not None

    foreign_evidence = run.evidence.model_copy(
        update={"evidence_ref": "evidence://foreign/probe/1"},
    )
    state = admit_external_effect_unknown(correlation_id=correlation_id)
    evaluator = _evaluator_for_variant("payment_completed_after_unknown", correlation_id)
    result = evaluate_external_effect_evidence(
        state=state,
        evidence=foreign_evidence,
        tenant_id="tenant-lab",
        contract_id=SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
        evaluator_strategy=evaluator,
    )

    assert result.outcome is EvidenceEvaluationOutcome.READY_FOR_DECISION
    assert result.rationale == "evidence_sufficient_for_resolution_planning"


def test_intergrax_has_no_import_dependency_on_payment_evaluator_plugin() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    intergrax_root = repo_root / "intergrax"
    needle = "payment_evidence_evaluator"
    violations: list[str] = []
    for path in intergrax_root.rglob("*.py"):
        if needle in path.read_text(encoding="utf-8"):
            violations.append(str(path.relative_to(repo_root)))
    assert not violations
