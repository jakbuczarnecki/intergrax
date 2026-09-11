# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""CRM enterprise preventive showcase analyzer (PREVENTIVE R6)."""

from __future__ import annotations

from intergrax.contracts.preventive.analyzer_descriptor import (
    PreventiveAnalyzerDescriptor,
    PreventiveResourceBudget,
)
from intergrax.contracts.preventive.category import PreventiveRecommendationCategory
from intergrax.contracts.preventive.context import PreventiveAnalysisInput
from intergrax.contracts.preventive.evidence import RecommendationEvidenceReference
from intergrax.contracts.preventive.recommendation import PreventiveRecommendationCandidate

_CRM_PREVENTIVE_DESCRIPTOR = PreventiveAnalyzerDescriptor(
    analyzer_id="crm_latency_preventive",
    namespace="intergrax.platform",
    version="crm_latency_preventive@1.0.0",
    owner="platform-diagnostics",
    capabilities=("crm_latency", "connector_retry_surge"),
    quality_profile_id="crm_latency_preventive",
    resource_budget=PreventiveResourceBudget(
        max_execution_time_ms=200,
        max_candidates_per_run=2,
    ),
)


class CrmLatencyPreventiveAnalyzer:
    analyzer_namespace = "intergrax.platform"
    analyzer_id = "crm_latency_preventive"
    analyzer_version = "crm_latency_preventive@1.0.0"
    priority = 100

    @property
    def descriptor(self) -> PreventiveAnalyzerDescriptor:
        return _CRM_PREVENTIVE_DESCRIPTOR

    def analyze(
        self,
        analysis_input: PreventiveAnalysisInput,
    ) -> tuple[PreventiveRecommendationCandidate, ...]:
        signal = analysis_input.risk_signal
        if signal.risk_type != "HIGH_LATENCY_RISK":
            return ()

        evidence: list[RecommendationEvidenceReference] = [
            RecommendationEvidenceReference(
                source_type="risk_signal",
                source_id=signal.signal_id,
                relation="supports",
            ),
        ]
        for ref in signal.evidence_refs:
            evidence.append(
                RecommendationEvidenceReference(
                    source_type="predictive_evidence",
                    source_id=ref,
                    relation="supports",
                ),
            )
        for ref in analysis_input.historical_outcome.similar_incident_refs:
            evidence.append(
                RecommendationEvidenceReference(
                    source_type="incident_history",
                    source_id=ref,
                    relation="similar_incident",
                ),
            )
        for ref in analysis_input.diagnostic_evidence.evidence_refs:
            evidence.append(
                RecommendationEvidenceReference(
                    source_type="diagnostic_evidence",
                    source_id=ref,
                    relation="context",
                ),
            )

        connector_retry_surge = any(
            "connector_retries" in ref for ref in analysis_input.diagnostic_evidence.evidence_refs
        )
        priority_label = "HIGH" if connector_retry_surge else "NORMAL"
        description = (
            "Validate external provider availability and payment connector latency"
            if connector_retry_surge
            else "Investigate payment connector latency and review API timeout configuration"
        )
        expected_impact = "Reduce probability of API timeout and CRM incident recurrence"

        reasoning = (
            f"Latency risk {signal.risk_type} at confidence {signal.confidence:.2f} "
            f"with {len(analysis_input.historical_outcome.similar_incident_refs)} prior incidents"
        )
        limitations: tuple[str, ...] = ("no external provider telemetry",)
        if not analysis_input.diagnostic_evidence.evidence_refs:
            limitations = (*limitations, "diagnostic evidence refs sparse")

        return (
            PreventiveRecommendationCandidate(
                category=PreventiveRecommendationCategory.INVESTIGATE,
                description=description,
                expected_impact=expected_impact,
                evidence_refs=tuple(evidence),
                analyzer_id=self.analyzer_id,
                analyzer_version=self.analyzer_version,
                raw_confidence=signal.confidence,
                priority_label=priority_label,
                reasoning_summary=reasoning,
                known_limitations=limitations,
            ),
        )


__all__ = ["CrmLatencyPreventiveAnalyzer"]
