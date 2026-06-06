# © Artur Czarnecki. All rights reserved.

"""Execution strategy tuning sub-engine (Phase W-ADAPT-2.3)."""

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


class ExecutionStrategyEngine:
    """Rule-based execution strategy proposals from step/retry/parallel metrics."""

    def __init__(
        self,
        *,
        max_delta_percent: float = 15.0,
        step_count_threshold: int = 12,
    ) -> None:
        self._max_delta_percent = max_delta_percent
        self._step_count_threshold = step_count_threshold

    @property
    def engine_id(self) -> str:
        return "execution_strategy"

    def propose(self, context: AdaptationEngineContext) -> list[AdaptationProposalCandidate]:
        if not context.signals:
            return []

        step_counts = [signal.step_count for signal in context.signals]
        average_steps = sum(step_counts) / len(step_counts)
        regression_hits = sum(
            1 for signal in context.signals if "step_explosion" in signal.regression_flags
        )
        if average_steps < self._step_count_threshold and regression_hits == 0:
            return []

        signal_id = context.signals[-1].signal_id
        envelope = AdaptiveLoopEnvelope(
            loop_id=f"execution-strategy-{context.task_class}",
            kind=AdaptiveLoopKind.EXECUTION_STRATEGY_TUNING,
            max_iterations=4,
            max_delta_percent=self._max_delta_percent,
            authority=AdaptiveAuthorityLevel.RECOMMEND,
            requires_human_approval=False,
            cooldown_seconds=1800,
        )
        proposal = AdaptiveLoopProposal(
            envelope=envelope,
            proposed_change_summary=(
                f"Tighten execution strategy for '{context.task_class}': "
                f"avg_steps={average_steps:.1f}, step_explosion_flags={regression_hits}"
            ),
            evaluation_signal_id=signal_id,
        )
        draft = ProfileVersionDraft(
            version_id=f"draft-orch-{context.task_class}",
            artifact_type=ProfileArtifactType.ORCHESTRATION,
            artifact_payload={
                "max_parallel_nodes": 2,
                "retry_policy_name": "strict",
                "max_delta_percent": self._max_delta_percent,
            },
            created_by=self.engine_id,
        )
        rank_score = min(1.0, average_steps / max(1, self._step_count_threshold))
        return [
            AdaptationProposalCandidate(
                loop_id=envelope.loop_id,
                source_engine=self.engine_id,
                proposal=proposal,
                profile_draft=draft,
                rank_score=rank_score,
                cooldown_seconds=envelope.cooldown_seconds,
            )
        ]
