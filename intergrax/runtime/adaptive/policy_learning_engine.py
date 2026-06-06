# © Artur Czarnecki. All rights reserved.

"""Policy learning sub-engine (Phase W-ADAPT-2.4)."""

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


class PolicyLearningEngine:
    """Policy-learning proposals requiring human approval (recommend-only)."""

    def __init__(self, *, max_delta_percent: float = 15.0) -> None:
        self._max_delta_percent = max_delta_percent

    @property
    def engine_id(self) -> str:
        return "policy_learning"

    def propose(self, context: AdaptationEngineContext) -> list[AdaptationProposalCandidate]:
        if not context.signals:
            return []

        risky_flags = {"tool_usage_drop", "llm_cost_spike"}
        flagged = [
            signal
            for signal in context.signals
            if any(flag in signal.regression_flags for flag in risky_flags)
        ]
        if not flagged:
            return []

        signal = flagged[-1]
        envelope = AdaptiveLoopEnvelope(
            loop_id=f"policy-learning-{context.task_class}",
            kind=AdaptiveLoopKind.POLICY_LEARNING,
            max_iterations=3,
            max_delta_percent=min(self._max_delta_percent, 25.0),
            authority=AdaptiveAuthorityLevel.AUTO_WITH_HUMAN_GATE,
            requires_human_approval=True,
            cooldown_seconds=7200,
        )
        proposal = AdaptiveLoopProposal(
            envelope=envelope,
            proposed_change_summary=(
                f"Recommend tool policy tightening for '{context.task_class}' "
                f"based on regression flags {signal.regression_flags}"
            ),
            human_approver_id=context.default_human_approver_id,
            evaluation_signal_id=signal.signal_id,
        )
        draft = ProfileVersionDraft(
            version_id=f"draft-policy-{context.task_class}",
            artifact_type=ProfileArtifactType.POLICY_FRAGMENT,
            artifact_payload={"deny_tool_ids": ["sandbox.exec"], "max_delta_percent": envelope.max_delta_percent},
            created_by=self.engine_id,
        )
        return [
            AdaptationProposalCandidate(
                loop_id=envelope.loop_id,
                source_engine=self.engine_id,
                proposal=proposal,
                profile_draft=draft,
                rank_score=0.85,
                cooldown_seconds=envelope.cooldown_seconds,
            )
        ]
