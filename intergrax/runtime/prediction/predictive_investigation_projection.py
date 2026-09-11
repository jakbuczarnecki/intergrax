# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Map predictive engine output to investigation read models (PREDICTIVE R1)."""

from __future__ import annotations

from intergrax.contracts.predictive_risk import PredictiveRiskSignal
from intergrax.contracts.predictive_investigation_read import RelatedPredictiveRiskSignalView


def project_related_risk_signals(
    signals: tuple[PredictiveRiskSignal, ...],
    *,
    analyzer_id_by_signal: dict[str, str] | None = None,
) -> tuple[RelatedPredictiveRiskSignalView, ...]:
    mapping = analyzer_id_by_signal or {}
    views: list[RelatedPredictiveRiskSignalView] = []
    for signal in signals:
        views.append(
            RelatedPredictiveRiskSignalView(
                signal_id=signal.signal_id,
                tenant_id=signal.tenant_id,
                scope=signal.scope,
                subject_identity=signal.subject_identity,
                risk_type=signal.risk_type,
                severity=signal.severity,
                confidence=signal.confidence,
                evidence_refs=signal.evidence_refs,
                prediction_window_label=signal.prediction_window.label,
                generated_at=signal.generated_at,
                model_version=signal.model_version,
                summary=signal.summary,
                recommended_actions=signal.recommended_actions,
                analyzer_id=mapping.get(
                    signal.signal_id,
                    signal.analyzer_metadata.analyzer_id,
                ),
            ),
        )
    return tuple(views)


__all__ = ["project_related_risk_signals"]
