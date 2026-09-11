# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Orchestrate preventive analyzers — recommendations and audit only (PREVENTIVE R6)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from intergrax.contracts.predictive.analyzer_quality_profile import PredictiveAnalyzerQualityProfile
from intergrax.contracts.predictive.context_quality import PredictiveContextQualityReport
from intergrax.contracts.predictive_risk import PredictiveRiskSignal
from intergrax.contracts.preventive.audit import (
    PreventiveAuditRecord,
    PreventiveRecommendationAuditRecord,
)
from intergrax.contracts.preventive.conflict import PreventiveRecommendationConflict
from intergrax.contracts.preventive.context import PreventiveAnalysisInput
from intergrax.contracts.preventive.governance import PreventiveRecommendationGovernance
from intergrax.contracts.preventive.lifecycle import PreventiveRecommendationLifecycleState
from intergrax.contracts.preventive.safety import preventive_safety_always_disabled
from intergrax.contracts.preventive.recommendation import (
    PreventiveRecommendation,
    PreventiveRecommendationCandidate,
    mint_preventive_recommendation_id,
)
from intergrax.runtime.prediction.governance.analyzer_quality_store import (
    InMemoryPredictiveAnalyzerQualityStore,
)
from intergrax.runtime.prediction.governance.predictive_context_quality_evaluator import (
    PredictiveContextQualityEvaluator,
)
from intergrax.runtime.prevention.preventive_confidence_evaluator import (
    PreventiveConfidenceEvaluator,
)
from intergrax.runtime.prevention.preventive_recommendation_validator import validate_candidate
from intergrax.runtime.prevention.preventive_evidence_qualification import evidence_quality_label
from intergrax.runtime.prevention.preventive_recommendation_conflict_resolver import (
    resolve_preventive_recommendation_conflicts,
)
from intergrax.runtime.prevention.preventive_registry import PreventiveAnalyzerRegistry

DEFAULT_PREVENTIVE_TIME_BUDGET_MS = 500
ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE = "PLUGIN_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class PreventiveIntelligenceEngineResult:
    recommendations: tuple[PreventiveRecommendation, ...]
    audit: PreventiveRecommendationAuditRecord
    recommendation_audit: tuple[PreventiveAuditRecord, ...]
    conflicts: tuple[PreventiveRecommendationConflict, ...]


@dataclass
class PreventiveIntelligenceEngine:
    registry: PreventiveAnalyzerRegistry
    time_budget_ms: int = DEFAULT_PREVENTIVE_TIME_BUDGET_MS
    quality_store: InMemoryPredictiveAnalyzerQualityStore = field(
        default_factory=InMemoryPredictiveAnalyzerQualityStore,
    )
    context_evaluator: PredictiveContextQualityEvaluator = field(
        default_factory=PredictiveContextQualityEvaluator,
    )
    confidence_evaluator: PreventiveConfidenceEvaluator = field(
        default_factory=PreventiveConfidenceEvaluator,
    )

    def recommend(self, analysis_input: PreventiveAnalysisInput) -> PreventiveIntelligenceEngineResult:
        tenant_id = analysis_input.predictive_context.tenant_id
        if tenant_id.strip() == "":
            raise ValueError("tenant_id required")

        recommendation_run_id = f"prr_{uuid4().hex}"
        deadline = time.monotonic() + (self.time_budget_ms / 1000.0)
        candidates: list[PreventiveRecommendationCandidate] = []
        outcomes: list[str] = []
        degraded = False
        recorded_at = datetime.now(tz=UTC)
        context_quality = self.context_evaluator.evaluate(analysis_input.predictive_context)

        for analyzer in self.registry.analyzers:
            if time.monotonic() > deadline:
                degraded = True
                outcomes.append(f"{analyzer.analyzer_id}:skipped_time_budget")
                break
            try:
                batch = analyzer.analyze(analysis_input)
            except Exception:
                degraded = True
                outcomes.append(f"{analyzer.analyzer_id}:{ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE}")
                continue
            for candidate in batch:
                if candidate.analyzer_id != analyzer.analyzer_id:
                    raise ValueError("candidate analyzer_id mismatch")
            candidates.extend(batch)
            outcomes.append(f"{analyzer.analyzer_id}:ok:{len(batch)}")

        recommendations = self._finalize_candidates(
            candidates=candidates,
            analysis_input=analysis_input,
            context_quality=context_quality,
            recorded_at=recorded_at,
        )
        rec_tuple = tuple(recommendations)
        conflicts = resolve_preventive_recommendation_conflicts(
            rec_tuple,
            scope_subject=analysis_input.risk_signal.subject_identity,
        )
        recommendation_audit = tuple(
            PreventiveAuditRecord(
                recommendation_id=item.recommendation_id,
                prediction_signal_id=item.risk_signal_id,
                context_snapshot_id=item.context_snapshot_id,
                analyzer_id=item.analyzer_id,
                analyzer_version=item.analyzer_version,
                evidence_refs=item.evidence_refs,
                confidence=item.confidence,
                created_at=item.created_at,
                prediction_run_id=item.prediction_run_id,
                tenant_id=item.tenant_id,
            )
            for item in rec_tuple
        )

        audit = PreventiveRecommendationAuditRecord(
            recommendation_run_id=recommendation_run_id,
            prediction_run_id=analysis_input.risk_signal.prediction_run_id,
            risk_signal_id=analysis_input.risk_signal.signal_id,
            tenant_id=tenant_id,
            recommendation_ids=tuple(item.recommendation_id for item in recommendations),
            analyzer_outcomes=tuple(outcomes),
            degraded=degraded,
            recorded_at=recorded_at,
        )
        return PreventiveIntelligenceEngineResult(
            recommendations=rec_tuple,
            audit=audit,
            recommendation_audit=recommendation_audit,
            conflicts=conflicts,
        )

    def _finalize_candidates(
        self,
        *,
        candidates: list[PreventiveRecommendationCandidate],
        analysis_input: PreventiveAnalysisInput,
        context_quality: PredictiveContextQualityReport,
        recorded_at: datetime,
    ) -> list[PreventiveRecommendation]:
        signal = analysis_input.risk_signal
        tenant_id = analysis_input.predictive_context.tenant_id
        finalized: list[PreventiveRecommendation] = []
        ordered = sorted(
            candidates,
            key=lambda c: (-c.raw_confidence, c.analyzer_id, c.category, c.description),
        )
        for candidate in ordered:
            validate_candidate(candidate)
            profile = self._profile_for(candidate.analyzer_id, tenant_id=tenant_id)
            confidence = self.confidence_evaluator.evaluate(
                risk_signal=signal,
                analyzer_profile=profile,
                historical_outcome=analysis_input.historical_outcome,
                context_quality=context_quality,
            )
            required_approval = (
                signal.severity.value in {"critical", "high"}
                or candidate.category in {"CONFIGURATION_REVIEW", "SECURITY_REVIEW"}
                or candidate.priority_label == "HIGH"
            )
            governance = PreventiveRecommendationGovernance(
                risk_level=signal.severity,
                confidence=confidence,
                required_approval=required_approval,
                execution_allowed=False,
                priority_label=candidate.priority_label,
            )
            safety = preventive_safety_always_disabled(
                requires_human_review=True,
                risk_level=signal.severity.value.upper(),
                governance_status="PENDING",
            )
            reasoning = candidate.reasoning_summary.strip() or candidate.description
            limitations = candidate.known_limitations or ("operator must validate scope",)
            finalized.append(
                PreventiveRecommendation(
                    recommendation_id=mint_preventive_recommendation_id(),
                    prediction_run_id=signal.prediction_run_id,
                    risk_signal_id=signal.signal_id,
                    tenant_id=tenant_id,
                    category=candidate.category,
                    description=candidate.description,
                    expected_impact=candidate.expected_impact,
                    confidence=confidence,
                    evidence_refs=candidate.evidence_refs,
                    governance=governance,
                    created_at=recorded_at,
                    analyzer_id=candidate.analyzer_id,
                    analyzer_version=candidate.analyzer_version,
                    safety=safety,
                    lifecycle_state=PreventiveRecommendationLifecycleState.VALIDATED.value,
                    reasoning_summary=reasoning,
                    known_limitations=limitations,
                    evidence_quality=evidence_quality_label(context_quality),
                    context_snapshot_id=analysis_input.predictive_context.context_snapshot_id,
                ),
            )
        return finalized

    def _profile_for(self, analyzer_id: str, *, tenant_id: str) -> PredictiveAnalyzerQualityProfile:
        return self.quality_store.get_profile(tenant_id=tenant_id, analyzer_id=analyzer_id)


__all__ = [
    "ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE",
    "DEFAULT_PREVENTIVE_TIME_BUDGET_MS",
    "PreventiveIntelligenceEngine",
    "PreventiveIntelligenceEngineResult",
]
