# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Investigation read projection for preventive recommendations (PREVENTIVE R6)."""

from __future__ import annotations

from intergrax.contracts.preventive.lifecycle import (
    PreventiveRecommendationLifecycleState,
    assert_lifecycle_transition,
)
from intergrax.contracts.preventive_investigation_read import RelatedPreventiveRecommendationView
from intergrax.contracts.preventive.recommendation import PreventiveRecommendation


def project_preventive_recommendations(
    recommendations: tuple[PreventiveRecommendation, ...],
    *,
    risk_type: str,
) -> tuple[RelatedPreventiveRecommendationView, ...]:
    views: list[RelatedPreventiveRecommendationView] = []
    for item in recommendations:
        current = PreventiveRecommendationLifecycleState(item.lifecycle_state)
        assert_lifecycle_transition(current, PreventiveRecommendationLifecycleState.PRESENTED)
        views.append(
            RelatedPreventiveRecommendationView(
                recommendation_id=item.recommendation_id,
                tenant_id=item.tenant_id,
                risk_signal_id=item.risk_signal_id,
                category=item.category,
                description=item.description,
                expected_impact=item.expected_impact,
                confidence=item.confidence,
                evidence_refs=item.evidence_refs,
                risk_level=item.governance.risk_level,
                required_approval=item.governance.required_approval,
                execution_allowed=item.governance.execution_allowed,
                priority_label=item.governance.priority_label,
                analyzer_id=item.analyzer_id,
                related_risk_type=risk_type,
                lifecycle_state=PreventiveRecommendationLifecycleState.PRESENTED.value,
                reasoning_summary=item.reasoning_summary,
                known_limitations=item.known_limitations,
                evidence_quality=item.evidence_quality,
            ),
        )
    return tuple(views)


__all__ = ["project_preventive_recommendations"]
