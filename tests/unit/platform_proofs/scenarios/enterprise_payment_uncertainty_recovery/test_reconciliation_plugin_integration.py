# © Artur Czarnecki. All rights reserved.

"""ERL reconciliation plugin boundary — scenario adapter + platform orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.contracts.enterprise_reliability import (
    EnterpriseReliabilityCapabilityKind,
    EnterpriseReliabilityPluginDescriptor,
    ExternalEffectEvidenceVerdict,
    ExternalEffectOutcome,
    ReconciliationExecutionDisposition,
    ReconciliationProbeRequest,
    ReconciliationProbeResult,
    ReconciliationStrategyAdvice,
    ResolutionPlatformAction,
    UncertaintyLifecyclePhase,
    UncertaintyResolutionKind,
)
from intergrax.contracts.enterprise_reliability.plugin_spi import (
    EnterpriseReliabilityStrategyContext,
    ResolutionStrategyEvaluationRequest,
)
from intergrax.contracts.enterprise_reliability.resolution_decision import ResolutionDecision
from intergrax.runtime.enterprise_reliability import (
    EnterpriseReliabilityPluginGatewayImpl,
    InMemoryEnterpriseReliabilityPluginRegistry,
    admit_external_effect_unknown_with_contract,
    execute_external_effect_reconciliation_probe,
    plan_external_effect_reconciliation,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.adapters.in_memory_external_reality_lookup import (
    InMemoryExternalRealityLookup,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.constants import (
    EXTERNAL_EFFECT_SOR_PROBE_REF,
    SCENARIO_RECONCILIATION_PLUGIN_ID,
    scenario_external_effect_contract,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.external_reality_lookup import (
    ExternalRealitySnapshot,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.wiring import (
    register_scenario_reconciliation_plugins,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.domain.sor_truth import (
    resolve_sor_truth_fields,
)

pytestmark = pytest.mark.unit

_SCENARIO_ROOT = (
    Path(__file__).resolve().parents[5]
    / "platform_proofs/scenarios/enterprise_payment_uncertainty_recovery"
)
_FIXED_TIME = datetime(2026, 9, 12, 12, 0, 0, tzinfo=UTC)


def _seed_lookup(variant_id: str, correlation_id: str) -> InMemoryExternalRealityLookup:
    variant_path = _SCENARIO_ROOT / "dataset/variants" / variant_id / "scenario_variant.json"
    document = json.loads(variant_path.read_text(encoding="utf-8"))
    fields = resolve_sor_truth_fields(document)
    lookup = InMemoryExternalRealityLookup()
    lookup.seed(
        ExternalRealitySnapshot(
            correlation_id=correlation_id,
            external_effect_reference="EXT-LAB",
            terminal_outcome=fields.terminal_outcome,
            funds_captured=fields.funds_captured,
            truth_availability_state=fields.truth_availability_state,
            sor_transaction_ref="SOR-LAB",
        ),
    )
    return lookup


def _run_reconciliation(
    lookup: InMemoryExternalRealityLookup,
    *,
    correlation_id: str = "corr-erl",
) -> tuple:
    contract = scenario_external_effect_contract()
    admission = admit_external_effect_unknown_with_contract(
        correlation_id=correlation_id,
        contract=contract,
    )
    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    register_scenario_reconciliation_plugins(registry, lookup)
    gateway = EnterpriseReliabilityPluginGatewayImpl(registry)
    planning = plan_external_effect_reconciliation(
        admission=admission,
        contract=contract,
        gateway=gateway,
        plugin_id=SCENARIO_RECONCILIATION_PLUGIN_ID,
        tenant_id="tenant-lab",
    )
    run = execute_external_effect_reconciliation_probe(
        planning=planning,
        gateway=gateway,
        tenant_id="tenant-lab",
        recorded_at=_FIXED_TIME,
    )
    return planning, run


@pytest.mark.parametrize(
    ("variant_id", "expected_verdict", "expected_outcome"),
    [
        (
            "payment_completed_after_unknown",
            ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS,
            ExternalEffectOutcome.SUCCESS,
        ),
        (
            "payment_failed_after_unknown",
            ExternalEffectEvidenceVerdict.DEFINITIVE_FAILURE,
            ExternalEffectOutcome.FAILURE,
        ),
        (
            "payment_truth_unavailable",
            ExternalEffectEvidenceVerdict.INSUFFICIENT,
            ExternalEffectOutcome.UNKNOWN,
        ),
    ],
)
def test_erl_probe_execution_via_scenario_plugin(
    variant_id: str,
    expected_verdict: ExternalEffectEvidenceVerdict,
    expected_outcome: ExternalEffectOutcome,
) -> None:
    correlation_id = f"corr-{variant_id}"
    lookup = _seed_lookup(variant_id, correlation_id)
    _planning, run = _run_reconciliation(lookup, correlation_id=correlation_id)

    assert run.execution.disposition is ReconciliationExecutionDisposition.PROBE_EXECUTED
    assert run.execution.probe_result is not None
    assert run.execution.probe_result.verdict is expected_verdict
    assert run.state.effect_outcome is expected_outcome
    assert run.evidence is not None
    assert run.evidence.operation_link.correlation_id == correlation_id
    assert run.evidence.operation_link.probe_ref == EXTERNAL_EFFECT_SOR_PROBE_REF
    if expected_outcome is ExternalEffectOutcome.UNKNOWN:
        assert run.state.lifecycle_phase is UncertaintyLifecyclePhase.PENDING_RESOLUTION
    else:
        assert run.state.lifecycle_phase is UncertaintyLifecyclePhase.RESOLVED
        assert run.state.resolution_kind in {
            UncertaintyResolutionKind.CONFIRMED_SUCCESS,
            UncertaintyResolutionKind.CONFIRMED_FAILURE,
        }


def test_source_unavailable_returns_insufficient_not_silent_unknown() -> None:
    lookup = InMemoryExternalRealityLookup()
    lookup.set_source_available(False)
    _planning, run = _run_reconciliation(lookup)
    assert run.execution.probe_result is not None
    assert run.execution.probe_result.verdict is ExternalEffectEvidenceVerdict.INSUFFICIENT
    assert "source_unavailable" in run.execution.probe_result.rationale
    assert run.state.effect_outcome is ExternalEffectOutcome.UNKNOWN


def test_record_missing_returns_insufficient_with_rationale() -> None:
    lookup = InMemoryExternalRealityLookup()
    _planning, run = _run_reconciliation(lookup, correlation_id="corr-missing")
    assert run.execution.probe_result is not None
    assert run.execution.probe_result.verdict is ExternalEffectEvidenceVerdict.INSUFFICIENT
    assert "record_missing" in run.execution.probe_result.rationale


@dataclass(frozen=True, slots=True)
class _ReplacementReconciliationPlugin:
    _descriptor: EnterpriseReliabilityPluginDescriptor

    @property
    def plugin_id(self) -> str:
        return self._descriptor.plugin_id

    @property
    def version(self) -> str:
        return self._descriptor.version

    @property
    def descriptor(self) -> EnterpriseReliabilityPluginDescriptor:
        return self._descriptor

    def evaluate(
        self,
        context: EnterpriseReliabilityStrategyContext,
    ) -> ReconciliationStrategyAdvice | None:
        return ReconciliationStrategyAdvice(probe_ref=EXTERNAL_EFFECT_SOR_PROBE_REF)

    def execute_probe(
        self,
        request: ReconciliationProbeRequest,
    ) -> ReconciliationProbeResult:
        return ReconciliationProbeResult(
            verdict=ExternalEffectEvidenceVerdict.DEFINITIVE_FAILURE,
            evidence_ref="evidence://replacement/probe",
            rationale="replacement_plugin",
        )


@dataclass(frozen=True, slots=True)
class _ReplacementResolutionPlugin:
    _descriptor: EnterpriseReliabilityPluginDescriptor

    @property
    def plugin_id(self) -> str:
        return self._descriptor.plugin_id

    @property
    def version(self) -> str:
        return self._descriptor.version

    @property
    def descriptor(self) -> EnterpriseReliabilityPluginDescriptor:
        return self._descriptor

    def evaluate(
        self,
        request: ResolutionStrategyEvaluationRequest,
    ) -> ResolutionDecision | None:
        return ResolutionDecision(
            action=ResolutionPlatformAction.CONTINUE,
            rationale="replacement_resolution",
        )


def test_plugin_replacement_changes_probe_outcome() -> None:
    lookup = _seed_lookup("payment_completed_after_unknown", "corr-replace")
    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    register_scenario_reconciliation_plugins(registry, lookup)
    replacement_id = "replacement-reconcile"
    registry.register(
        _ReplacementReconciliationPlugin(
            EnterpriseReliabilityPluginDescriptor(
                plugin_id=replacement_id,
                version="9.9.9",
                owner="test",
                capability_kind=EnterpriseReliabilityCapabilityKind.RECONCILIATION,
                capabilities=("replace",),
                tenant_scope=None,
                priority=10,
            ),
        ),
    )
    registry.register(
        _ReplacementResolutionPlugin(
            EnterpriseReliabilityPluginDescriptor(
                plugin_id=replacement_id,
                version="9.9.9",
                owner="test",
                capability_kind=EnterpriseReliabilityCapabilityKind.RESOLUTION,
                capabilities=("resolve",),
                tenant_scope=None,
                priority=10,
            ),
        ),
    )
    contract = scenario_external_effect_contract()
    admission = admit_external_effect_unknown_with_contract(
        correlation_id="corr-replace",
        contract=contract,
    )
    gateway = EnterpriseReliabilityPluginGatewayImpl(registry)
    planning = plan_external_effect_reconciliation(
        admission=admission,
        contract=contract,
        gateway=gateway,
        plugin_id=replacement_id,
        tenant_id="tenant-lab",
    )
    run = execute_external_effect_reconciliation_probe(
        planning=planning,
        gateway=gateway,
        tenant_id="tenant-lab",
    )
    assert run.execution.probe_result is not None
    assert run.execution.probe_result.rationale == "replacement_plugin"
    assert run.state.effect_outcome is ExternalEffectOutcome.FAILURE


def test_intergrax_enterprise_reliability_contracts_have_no_payment_domain_leaks() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    contracts_root = repo_root / "intergrax/contracts/enterprise_reliability"
    forbidden = ("payment_intent", "PAYMENT_COMPLETED", "postgresql", "acquirer")
    violations: list[str] = []
    for path in contracts_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            if token.lower() in text:
                violations.append(f"{path.name}:{token}")
    assert not violations, f"domain leak in ERL contracts: {violations}"
