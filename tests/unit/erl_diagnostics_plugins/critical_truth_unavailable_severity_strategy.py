# © Artur Czarnecki. All rights reserved.

"""External severity plugin — TRUTH_UNAVAILABLE → CRITICAL (ERL-DIAG-001D acceptance)."""

from __future__ import annotations

from intergrax.contracts.enterprise_reliability.diagnostics import (
    ExternalEffectReliabilityDiagnosticClassificationContext,
    ExternalEffectReliabilityDiagnosticSeverity,
    ExternalEffectReliabilitySeverityDecision,
    ReliabilityDiagnosticSeverityStrategyId,
    ReliabilityDiagnosticSeverityStrategyVersion,
)
from intergrax.contracts.enterprise_reliability.diagnostics.taxonomy import (
    ExternalEffectReliabilitySignalKind,
)

_PLUGIN_SEVERITY_STRATEGY_ID = ReliabilityDiagnosticSeverityStrategyId(
    "test.external.erl_diagnostics_plugins.critical_truth_unavailable.v1",
)
_PLUGIN_SEVERITY_STRATEGY_VERSION = ReliabilityDiagnosticSeverityStrategyVersion("1")


class CriticalTruthUnavailableSeverityStrategy:
    @property
    def strategy_id(self) -> ReliabilityDiagnosticSeverityStrategyId:
        return _PLUGIN_SEVERITY_STRATEGY_ID

    @property
    def strategy_version(self) -> ReliabilityDiagnosticSeverityStrategyVersion:
        return _PLUGIN_SEVERITY_STRATEGY_VERSION

    def classify(
        self,
        context: ExternalEffectReliabilityDiagnosticClassificationContext,
    ) -> ExternalEffectReliabilitySeverityDecision:
        if context.signal_kind is ExternalEffectReliabilitySignalKind.TRUTH_UNAVAILABLE:
            severity = ExternalEffectReliabilityDiagnosticSeverity.CRITICAL
            reason_code = "plugin_truth_unavailable_critical"
        else:
            severity = ExternalEffectReliabilityDiagnosticSeverity.WARNING
            reason_code = "plugin_default_warning"
        return ExternalEffectReliabilitySeverityDecision(
            severity=severity,
            strategy_id=self.strategy_id,
            strategy_version=self.strategy_version,
            reason_code=reason_code,
            safe_explanation="External test plugin severity classification.",
        )


__all__ = ["CriticalTruthUnavailableSeverityStrategy"]
