# © Artur Czarnecki. All rights reserved.

"""Evaluation feedback sub-engine (Phase W-ADAPT-2.5)."""

from __future__ import annotations

from intergrax.runtime.adaptive.adaptation_models import (
    AdaptationEngineContext,
    AdaptationProposalCandidate,
)
from intergrax.runtime.architecture.adaptive_governance import (
    AdaptiveAuthorityLevel,
    AdaptiveLoopEnvelope,
    AdaptiveLoopKind,
    AdaptiveLoopProposal,
)


class EvaluationFeedbackEngine:
    """Observe-only evaluation feedback proposals from registry trends."""

    @property
    def engine_id(self) -> str:
        return "evaluation_feedback"

    def propose(self, context: AdaptationEngineContext) -> list[AdaptationProposalCandidate]:
        trend = context.evaluation_trend
        if trend is None or not trend.comparisons:
            return []

        latest = trend.comparisons[-1]
        if latest.delta >= 0.0:
            return []

        signal_id = context.signals[-1].signal_id if context.signals else None
        envelope = AdaptiveLoopEnvelope(
            loop_id=f"evaluation-feedback-{context.task_class}",
            kind=AdaptiveLoopKind.EVALUATION_FEEDBACK,
            max_iterations=20,
            max_delta_percent=5.0,
            authority=AdaptiveAuthorityLevel.OBSERVE_ONLY,
            requires_human_approval=False,
            cooldown_seconds=900,
        )
        proposal = AdaptiveLoopProposal(
            envelope=envelope,
            proposed_change_summary=(
                f"Benchmark regression detected ({latest.release_from} -> {latest.release_to}, "
                f"delta={latest.delta:.3f}); trigger re-eval observe-only"
            ),
            evaluation_signal_id=signal_id,
        )
        return [
            AdaptationProposalCandidate(
                loop_id=envelope.loop_id,
                source_engine=self.engine_id,
                proposal=proposal,
                profile_draft=None,
                rank_score=abs(latest.delta),
                cooldown_seconds=envelope.cooldown_seconds,
            )
        ]
