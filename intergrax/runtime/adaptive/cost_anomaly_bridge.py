# © Artur Czarnecki. All rights reserved.

"""Cost anomaly to routing proposal bridge (Phase W-ADAPT-2.10)."""

from __future__ import annotations

from intergrax.runtime.adaptive.adaptation_models import (
    AdaptationEngineContext,
    AdaptationProposalCandidate,
)
from intergrax.runtime.adaptive.contracts import ProfileArtifactType, ProfileVersionDraft
from intergrax.runtime.architecture.adaptive_governance import (
    AdaptiveAuthorityLevel,
    AdaptiveLoopEnvelope,
    AdaptiveLoopKind,
    AdaptiveLoopProposal,
)
from intergrax.runtime.architecture.cost_forecast import CostAnomalyRecord, CostAnomalySeverity
from intergrax.runtime.architecture.cost_optimization import (
    OptimizationGuardrail,
    build_cost_optimization_report,
)


def proposals_from_cost_anomalies(
    context: AdaptationEngineContext,
) -> list[AdaptationProposalCandidate]:
    """Convert cost anomalies into routing/cost recommendation candidates."""
    if not context.cost_anomalies:
        return []

    report = build_cost_optimization_report(
        anomalies=context.cost_anomalies,
        guardrails=[
            OptimizationGuardrail(
                guardrail_id="default-savings-cap",
                description="Cap recommended savings ratio for adaptive proposals",
                max_recommended_savings_ratio=0.30,
            )
        ],
    )
    candidates: list[AdaptationProposalCandidate] = []
    for recommendation in report.recommendations:
        if not recommendation.policy_compliant:
            continue
        matching = _find_anomaly(context.cost_anomalies, recommendation.scope_id)
        if matching is None:
            continue
        signal_id = context.signals[-1].signal_id if context.signals else None
        envelope = AdaptiveLoopEnvelope(
            loop_id=f"cost-routing-{recommendation.scope_id}",
            kind=AdaptiveLoopKind.ROUTING_TUNING,
            max_iterations=3,
            max_delta_percent=10.0,
            authority=AdaptiveAuthorityLevel.RECOMMEND,
            requires_human_approval=False,
            cooldown_seconds=3600,
        )
        proposal = AdaptiveLoopProposal(
            envelope=envelope,
            proposed_change_summary=(
                f"Cost anomaly ({matching.severity.value}) for scope '{recommendation.scope_id}': "
                f"recommend {recommendation.recommendation_type.value}"
            ),
            evaluation_signal_id=signal_id,
        )
        draft = ProfileVersionDraft(
            version_id=f"draft-cost-{recommendation.scope_id}",
            artifact_type=ProfileArtifactType.LLM_ROUTING,
            artifact_payload={
                "recommendation_type": recommendation.recommendation_type.value,
                "estimated_savings_ratio": recommendation.estimated_savings_ratio,
            },
            created_by="cost_anomaly_bridge",
        )
        severity_rank = 0.5 if matching.severity == CostAnomalySeverity.WARNING else 0.9
        candidates.append(
            AdaptationProposalCandidate(
                loop_id=envelope.loop_id,
                source_engine="cost_anomaly_bridge",
                proposal=proposal,
                profile_draft=draft,
                rank_score=severity_rank,
                cooldown_seconds=envelope.cooldown_seconds,
            )
        )
    return candidates


def _find_anomaly(
    anomalies: list[CostAnomalyRecord],
    scope_id: str,
) -> CostAnomalyRecord | None:
    for anomaly in anomalies:
        if anomaly.scope_id == scope_id:
            return anomaly
    return None
