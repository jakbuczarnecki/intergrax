# © Artur Czarnecki. All rights reserved.

"""Payment resolution strategy plugin — SPI, rules, and orchestration integration."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.contracts.enterprise_reliability import (
    EvidenceEvaluationOutcome,
    ExternalEffectEvidenceVerdict,
    ResolutionDisposition,
    ResolutionPlatformAction,
    ResolutionStrategy,
    UnknownUncertaintyPosture,
)
from intergrax.contracts.enterprise_reliability.plugin_spi import (
    EnterpriseReliabilityStrategyContext,
    ResolutionStrategyEvaluationRequest,
)
from intergrax.runtime.enterprise_reliability import (
    admit_external_effect_unknown,
    evaluate_external_effect_evidence,
    plan_external_effect_resolution,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.adapters.in_memory_payment_evidence_lookup import (
    InMemoryPaymentReconciliationEvidenceLookup,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.constants import (
    SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
    SCENARIO_RECONCILIATION_PLUGIN_ID,
    scenario_external_effect_contract,
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
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.mapping.payment_resolution_decision import (
    decide_payment_resolution,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.plugins.payment_evidence_evaluator import (
    PaymentEvidenceEvaluatorPlugin,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.plugins.payment_resolution_strategy import (
    PaymentResolutionStrategyPlugin,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.wiring import (
    register_scenario_reconciliation_plugins,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.domain.sor_truth import (
    resolve_sor_truth_fields,
)
from intergrax.runtime.enterprise_reliability import (
    EnterpriseReliabilityPluginGatewayImpl,
    InMemoryEnterpriseReliabilityPluginRegistry,
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


def _payment_lookup_for_variant(
    variant_id: str,
    correlation_id: str,
) -> InMemoryPaymentReconciliationEvidenceLookup:
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
    return payment_lookup


def _resolution_plugin_for_variant(
    variant_id: str,
    correlation_id: str,
) -> PaymentResolutionStrategyPlugin:
    return PaymentResolutionStrategyPlugin(_lookup=_payment_lookup_for_variant(variant_id, correlation_id))


def test_successful_payment_evidence_yields_continue_resolution() -> None:
    correlation_id = "corr-res-success"
    variant_id = "payment_completed_after_unknown"
    lookup = _seed_lookup(variant_id, correlation_id)
    _planning, run = _run_reconciliation(lookup, correlation_id=correlation_id)
    assert run.evidence is not None

    plugin = _resolution_plugin_for_variant(variant_id, correlation_id)
    request = ResolutionStrategyEvaluationRequest(
        evidence=run.evidence,
        execution_context=EnterpriseReliabilityStrategyContext(
            tenant_id="tenant-lab",
            correlation_id=correlation_id,
            contract_id=SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
            effect_outcome=run.state.effect_outcome,
            lifecycle_phase=run.state.lifecycle_phase,
            evidence_verdict=run.evidence.verdict,
            evidence_ref=run.evidence.evidence_ref,
        ),
        effect_contract=scenario_external_effect_contract(),
    )
    decision = plugin.evaluate(request)
    assert decision is not None
    assert decision.action is ResolutionPlatformAction.CONTINUE
    assert "continue" in decision.rationale


def test_failed_payment_evidence_yields_stop_resolution() -> None:
    correlation_id = "corr-res-failed"
    variant_id = "payment_failed_after_unknown"
    lookup = _seed_lookup(variant_id, correlation_id)
    _planning, run = _run_reconciliation(lookup, correlation_id=correlation_id)
    assert run.evidence is not None
    assert run.evidence.verdict is ExternalEffectEvidenceVerdict.DEFINITIVE_FAILURE

    plugin = _resolution_plugin_for_variant(variant_id, correlation_id)
    request = ResolutionStrategyEvaluationRequest(
        evidence=run.evidence,
        execution_context=EnterpriseReliabilityStrategyContext(
            tenant_id="tenant-lab",
            correlation_id=correlation_id,
            contract_id=SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
            effect_outcome=run.state.effect_outcome,
            lifecycle_phase=run.state.lifecycle_phase,
            evidence_verdict=run.evidence.verdict,
            evidence_ref=run.evidence.evidence_ref,
        ),
        effect_contract=scenario_external_effect_contract(),
    )
    decision = plugin.evaluate(request)
    assert decision is not None
    assert decision.action is ResolutionPlatformAction.STOP


def test_unavailable_truth_yields_escalation_resolution() -> None:
    correlation_id = "corr-res-unavailable"
    variant_id = "payment_truth_unavailable"
    lookup = _seed_lookup(variant_id, correlation_id)
    _planning, run = _run_reconciliation(lookup, correlation_id=correlation_id)
    assert run.evidence is not None
    assert run.evidence.verdict is ExternalEffectEvidenceVerdict.INSUFFICIENT

    payment = _seed_payment_evidence(
        variant_id,
        correlation_id,
        snapshot=ExternalRealitySnapshot(
            correlation_id=correlation_id,
            external_effect_reference="EXT-LAB",
            terminal_outcome="TRUTH_INDETERMINATE",
            funds_captured=False,
            truth_availability_state="UNAVAILABLE",
            sor_transaction_ref=None,
        ),
    )
    decision = decide_payment_resolution(
        evidence=run.evidence,
        payment=payment,
        tenant_id="tenant-lab",
        contract_id=SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
        correlation_id=correlation_id,
    )
    assert decision.action is ResolutionPlatformAction.ESCALATE
    assert "unavailable" in decision.rationale


def test_plan_resolution_via_gateway_with_payment_bundle() -> None:
    correlation_id = "corr-res-plan"
    variant_id = "payment_completed_after_unknown"
    reality_lookup = _seed_lookup(variant_id, correlation_id)
    payment_lookup = _payment_lookup_for_variant(variant_id, correlation_id)
    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    register_scenario_reconciliation_plugins(
        registry,
        reality_lookup,
        payment_evidence_lookup=payment_lookup,
    )
    gateway = EnterpriseReliabilityPluginGatewayImpl(registry)
    _planning, run = _run_reconciliation(reality_lookup, correlation_id=correlation_id)
    assert run.evidence is not None

    state = admit_external_effect_unknown(correlation_id=correlation_id)
    evaluator = PaymentEvidenceEvaluatorPlugin(_lookup=payment_lookup)
    evidence_eval = evaluate_external_effect_evidence(
        state=state,
        evidence=run.evidence,
        tenant_id="tenant-lab",
        contract_id=SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
        evaluator_strategy=evaluator,
    )
    assert evidence_eval.outcome is EvidenceEvaluationOutcome.READY_FOR_DECISION

    planning = plan_external_effect_resolution(
        state=state,
        contract_id=SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
        effect_contract=scenario_external_effect_contract(),
        unknown_posture=UnknownUncertaintyPosture.RECONCILE_ONLY,
        evidence=run.evidence,
        gateway=gateway,
        plugin_id=SCENARIO_RECONCILIATION_PLUGIN_ID,
        tenant_id="tenant-lab",
    )
    assert planning.plan.disposition is ResolutionDisposition.INVOKE_PLUGIN
    assert planning.plan.decision is not None
    assert planning.plan.decision.action is ResolutionPlatformAction.CONTINUE


def test_plugin_implements_resolution_strategy_protocol() -> None:
    lookup = InMemoryPaymentReconciliationEvidenceLookup()
    plugin = PaymentResolutionStrategyPlugin(_lookup=lookup)
    assert isinstance(plugin, ResolutionStrategy)


def test_plugin_abstains_for_non_scenario_evidence_refs() -> None:
    correlation_id = "corr-res-abstain"
    variant_id = "payment_completed_after_unknown"
    lookup = _seed_lookup(variant_id, correlation_id)
    _planning, run = _run_reconciliation(lookup, correlation_id=correlation_id)
    assert run.evidence is not None

    foreign_evidence = run.evidence.model_copy(
        update={"evidence_ref": "evidence://foreign/probe/1"},
    )
    plugin = _resolution_plugin_for_variant(variant_id, correlation_id)
    request = ResolutionStrategyEvaluationRequest(
        evidence=foreign_evidence,
        execution_context=EnterpriseReliabilityStrategyContext(
            tenant_id="tenant-lab",
            correlation_id=correlation_id,
            contract_id=SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
            effect_outcome=run.state.effect_outcome,
            lifecycle_phase=run.state.lifecycle_phase,
            evidence_verdict=foreign_evidence.verdict,
            evidence_ref=foreign_evidence.evidence_ref,
        ),
        effect_contract=scenario_external_effect_contract(),
    )
    assert plugin.evaluate(request) is None


def test_intergrax_has_no_import_dependency_on_payment_resolution_plugin() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    intergrax_root = repo_root / "intergrax"
    needle = "payment_resolution_strategy"
    violations: list[str] = []
    for path in intergrax_root.rglob("*.py"):
        if needle in path.read_text(encoding="utf-8"):
            violations.append(str(path.relative_to(repo_root)))
    assert not violations
