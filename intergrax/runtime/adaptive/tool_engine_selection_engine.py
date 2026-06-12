# © Artur Czarnecki. All rights reserved.

"""Dynamic tool engine mode proposals (TOOL-ENG-10 · AUDIT-IDEAL routing)."""

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


class ToolEngineSelectionEngine:
    """Rule-based tool_selection_mode / tool_invocation_mode adaptation proposals."""

    def __init__(self, *, utility_threshold: float = 0.45) -> None:
        self._utility_threshold = utility_threshold

    @property
    def engine_id(self) -> str:
        return "tool_engine_selection"

    def propose(self, context: AdaptationEngineContext) -> list[AdaptationProposalCandidate]:
        if not context.signals:
            return []
        utilities = [signal.utility for signal in context.signals if signal.utility is not None]
        average_utility = sum(utilities) / len(utilities) if utilities else 0.0
        if average_utility >= self._utility_threshold:
            return []

        signal_id = context.signals[-1].signal_id
        envelope = AdaptiveLoopEnvelope(
            loop_id=f"tool-engine-{context.task_class}",
            kind=AdaptiveLoopKind.ROUTING_TUNING,
            max_iterations=3,
            max_delta_percent=10.0,
            authority=AdaptiveAuthorityLevel.RECOMMEND,
            requires_human_approval=False,
            cooldown_seconds=900,
        )
        proposal = AdaptiveLoopProposal(
            envelope=envelope,
            proposed_change_summary=(
                "Recommend semantic tool selection for large-catalog task classes"
            ),
            evaluation_signal_id=signal_id,
        )
        draft = ProfileVersionDraft(
            version_id=f"draft-tool-engine-{context.task_class}",
            artifact_type=ProfileArtifactType.ORCHESTRATION,
            artifact_payload={
                "tool_selection_mode": "semantic",
                "tool_invocation_mode": "single_pass",
            },
            created_by=self.engine_id,
        )
        return [
            AdaptationProposalCandidate(
                loop_id=envelope.loop_id,
                source_engine=self.engine_id,
                proposal=proposal,
                profile_draft=draft,
                rank_score=min(1.0, self._utility_threshold - average_utility + 0.1),
                cooldown_seconds=envelope.cooldown_seconds,
            )
        ]
