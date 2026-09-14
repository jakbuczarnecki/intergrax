# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Conservative default ERL reliability severity strategy (ERL-DIAG-001D)."""

from __future__ import annotations

from intergrax.contracts.enterprise_reliability.diagnostics.classification import (
    RELIABILITY_CONSERVATIVE_SEVERITY_STRATEGY_ID,
    RELIABILITY_CONSERVATIVE_SEVERITY_STRATEGY_VERSION,
    ExternalEffectReliabilityDiagnosticClassificationContext,
    ExternalEffectReliabilityDiagnosticSeverity,
    ExternalEffectReliabilitySeverityDecision,
    ReliabilityDiagnosticSeverityStrategyId,
    ReliabilityDiagnosticSeverityStrategyVersion,
)
from intergrax.contracts.enterprise_reliability.diagnostics.taxonomy import (
    AutomationSafetyHint,
    ExternalEffectReliabilitySignalKind,
)


class ConservativeReliabilitySeverityStrategy:
    """
    Domain-neutral default: factual signal posture and automation-safety hints only.

    Replaceable via composition — not customer business-criticality inference.
    """

    @property
    def strategy_id(self) -> ReliabilityDiagnosticSeverityStrategyId:
        return RELIABILITY_CONSERVATIVE_SEVERITY_STRATEGY_ID

    @property
    def strategy_version(self) -> ReliabilityDiagnosticSeverityStrategyVersion:
        return RELIABILITY_CONSERVATIVE_SEVERITY_STRATEGY_VERSION

    def classify(
        self,
        context: ExternalEffectReliabilityDiagnosticClassificationContext,
    ) -> ExternalEffectReliabilitySeverityDecision:
        if type(context) is not ExternalEffectReliabilityDiagnosticClassificationContext:
            raise TypeError("context must be ExternalEffectReliabilityDiagnosticClassificationContext")
        signal_kind = context.signal_kind
        safety = context.automation_safety_hint
        severity, reason_code = _conservative_severity_for_facts(signal_kind, safety)
        return ExternalEffectReliabilitySeverityDecision(
            severity=severity,
            strategy_id=self.strategy_id,
            strategy_version=self.strategy_version,
            reason_code=reason_code,
            safe_explanation=(
                "Conservative platform default severity from reliability signal facts."
            ),
        )


def _conservative_severity_for_facts(
    signal_kind: ExternalEffectReliabilitySignalKind,
    safety: AutomationSafetyHint,
) -> tuple[ExternalEffectReliabilityDiagnosticSeverity, str]:
    if signal_kind is ExternalEffectReliabilitySignalKind.AUTOMATION_SAFETY_LIMIT:
        return ExternalEffectReliabilityDiagnosticSeverity.ERROR, "automation_safety_limit"
    if signal_kind is ExternalEffectReliabilitySignalKind.TRUTH_UNAVAILABLE:
        base = ExternalEffectReliabilityDiagnosticSeverity.WARNING
        return _elevate_for_unsafe_automation(base, safety, "truth_unavailable")
    if signal_kind in (
        ExternalEffectReliabilitySignalKind.UNCERTAINTY_ADMITTED,
        ExternalEffectReliabilitySignalKind.EVIDENCE_INSUFFICIENT,
    ):
        base = ExternalEffectReliabilityDiagnosticSeverity.WARNING
        return _elevate_for_unsafe_automation(base, safety, signal_kind.value.lower())
    if signal_kind is ExternalEffectReliabilitySignalKind.TRUTH_ESTABLISHED:
        return ExternalEffectReliabilityDiagnosticSeverity.INFO, "truth_established"
    if signal_kind in (
        ExternalEffectReliabilitySignalKind.GOVERNANCE_POSTURE,
        ExternalEffectReliabilitySignalKind.RECOVERY_POSTURE,
        ExternalEffectReliabilitySignalKind.RESOLUTION_POSTURE,
    ):
        base = ExternalEffectReliabilityDiagnosticSeverity.WARNING
        return _elevate_for_unsafe_automation(base, safety, signal_kind.value.lower())
    if signal_kind in (
        ExternalEffectReliabilitySignalKind.RECONCILIATION_ATTEMPTED,
        ExternalEffectReliabilitySignalKind.EVIDENCE_SUFFICIENT,
    ):
        return ExternalEffectReliabilityDiagnosticSeverity.INFO, signal_kind.value.lower()
    base = ExternalEffectReliabilityDiagnosticSeverity.WARNING
    return _elevate_for_unsafe_automation(base, safety, "reliability_signal")


def _elevate_for_unsafe_automation(
    base: ExternalEffectReliabilityDiagnosticSeverity,
    safety: AutomationSafetyHint,
    reason_code: str,
) -> tuple[ExternalEffectReliabilityDiagnosticSeverity, str]:
    if safety is AutomationSafetyHint.UNSAFE:
        if base is ExternalEffectReliabilityDiagnosticSeverity.INFO:
            return ExternalEffectReliabilityDiagnosticSeverity.ERROR, f"{reason_code}_unsafe_automation"
        if base is ExternalEffectReliabilityDiagnosticSeverity.WARNING:
            return ExternalEffectReliabilityDiagnosticSeverity.ERROR, f"{reason_code}_unsafe_automation"
    return base, reason_code


__all__ = ["ConservativeReliabilitySeverityStrategy"]
