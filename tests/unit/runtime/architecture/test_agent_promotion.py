from __future__ import annotations

from intergrax.runtime.architecture.agent_certification import AgentCertificationEvaluation
from intergrax.runtime.architecture.agent_promotion import (
    PromotionEvidenceBundle,
    PromotionStage,
    evaluate_agent_promotion,
)


def test_promotion_fails_without_required_evidence() -> None:
    decision = evaluate_agent_promotion(
        PromotionEvidenceBundle(
            agent_id="agent:research",
            agent_version="1.0.0",
            source_stage=PromotionStage.DEV,
            target_stage=PromotionStage.STAGING,
            certification=AgentCertificationEvaluation(
                agent_id="agent:research",
                agent_version="1.0.0",
                eligible=False,
                reasons=["cert failed"],
            ),
        )
    )
    assert decision.approved is False
    assert any("not eligible" in reason for reason in decision.reasons)


def test_promotion_passes_with_complete_bundle() -> None:
    decision = evaluate_agent_promotion(
        PromotionEvidenceBundle(
            agent_id="agent:research",
            agent_version="1.0.0",
            source_stage=PromotionStage.DEV,
            target_stage=PromotionStage.STAGING,
            certification=AgentCertificationEvaluation(
                agent_id="agent:research",
                agent_version="1.0.0",
                eligible=True,
                reasons=[],
            ),
            evaluation_report_refs=["build/unified_evaluation_report.json"],
            rollback_plan_ref="runbook/research_rollback.md",
            change_ticket_ref="CHG-1",
        )
    )
    assert decision.approved is True
    assert decision.reasons == []
