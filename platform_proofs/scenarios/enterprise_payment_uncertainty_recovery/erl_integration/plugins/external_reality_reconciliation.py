"""Scenario reconciliation plugin — ``ReconciliationStrategy`` + ``ReconciliationProbeExecutor``."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.enterprise_reliability.evidence import ExternalEffectEvidenceVerdict
from intergrax.contracts.enterprise_reliability.plugin_spi import (
    EnterpriseReliabilityCapabilityKind,
    EnterpriseReliabilityPluginDescriptor,
    EnterpriseReliabilityStrategyContext,
    ReconciliationStrategyAdvice,
)
from intergrax.contracts.enterprise_reliability.reconciliation_execution import (
    ReconciliationProbeRequest,
    ReconciliationProbeResult,
)

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.constants import (
    EXTERNAL_EFFECT_SOR_PROBE_REF,
    SCENARIO_RECONCILIATION_PLUGIN_ID,
    SCENARIO_RECONCILIATION_PLUGIN_OWNER,
    SCENARIO_RECONCILIATION_PLUGIN_VERSION,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.external_reality_lookup import (
    ExternalRealityLookupPort,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.failures import (
    ExternalRealityInconsistentState,
    ExternalRealityLookupError,
    ExternalRealityRecordMissing,
    ExternalRealitySourceUnavailable,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.mapping.probe_result import (
    map_snapshot_to_probe_result,
)


def _descriptor() -> EnterpriseReliabilityPluginDescriptor:
    return EnterpriseReliabilityPluginDescriptor(
        plugin_id=SCENARIO_RECONCILIATION_PLUGIN_ID,
        version=SCENARIO_RECONCILIATION_PLUGIN_VERSION,
        owner=SCENARIO_RECONCILIATION_PLUGIN_OWNER,
        capability_kind=EnterpriseReliabilityCapabilityKind.RECONCILIATION,
        capabilities=("external_reality_probe",),
        tenant_scope=None,
        priority=0,
    )


def _insufficient_probe_result(
    request: ReconciliationProbeRequest,
    *,
    rationale: str,
) -> ReconciliationProbeResult:
    return ReconciliationProbeResult(
        verdict=ExternalEffectEvidenceVerdict.INSUFFICIENT,
        evidence_ref=(
            f"evidence://erl-qual-004/reconcile/"
            f"{request.correlation_id}/{request.attempt_index}/lookup_failed"
        ),
        rationale=rationale,
    )


@dataclass(frozen=True, slots=True)
class ScenarioExternalRealityReconciliationPlugin:
    """Domain-neutral probe executor backed by scenario external-reality lookup."""

    _lookup: ExternalRealityLookupPort

    @property
    def plugin_id(self) -> str:
        return SCENARIO_RECONCILIATION_PLUGIN_ID

    @property
    def version(self) -> str:
        return SCENARIO_RECONCILIATION_PLUGIN_VERSION

    @property
    def descriptor(self) -> EnterpriseReliabilityPluginDescriptor:
        return _descriptor()

    def evaluate(
        self,
        context: EnterpriseReliabilityStrategyContext,
    ) -> ReconciliationStrategyAdvice | None:
        return ReconciliationStrategyAdvice(
            probe_ref=EXTERNAL_EFFECT_SOR_PROBE_REF,
            rationale="scenario_external_reality_default_probe",
        )

    def execute_probe(
        self,
        request: ReconciliationProbeRequest,
    ) -> ReconciliationProbeResult:
        try:
            snapshot = self._lookup.lookup_by_correlation_id(request.correlation_id)
        except ExternalRealitySourceUnavailable as exc:
            return _insufficient_probe_result(request, rationale=f"source_unavailable:{exc}")
        except ExternalRealityRecordMissing as exc:
            return _insufficient_probe_result(request, rationale=f"record_missing:{exc}")
        except ExternalRealityLookupError as exc:
            return _insufficient_probe_result(request, rationale=f"lookup_error:{exc}")

        try:
            return map_snapshot_to_probe_result(request=request, snapshot=snapshot)
        except ExternalRealityInconsistentState as exc:
            return _insufficient_probe_result(request, rationale=str(exc))
