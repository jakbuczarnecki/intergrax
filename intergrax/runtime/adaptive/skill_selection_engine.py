# © Artur Czarnecki. All rights reserved.

"""Dynamic skill bundle selection sub-engine (AUDIT-IDEAL-12.2)."""

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


class SkillSelectionEngine:
    """Rule-based skill bundle proposals from low-utility task-class signals."""

    def __init__(
        self,
        *,
        candidate_bundles: tuple[str, ...] = ("rag", "workspace", "memory"),
        utility_threshold: float = 0.45,
        max_delta_percent: float = 10.0,
    ) -> None:
        self._candidate_bundles = candidate_bundles
        self._utility_threshold = utility_threshold
        self._max_delta_percent = max_delta_percent

    @property
    def engine_id(self) -> str:
        return "skill_selection"

    def propose(self, context: AdaptationEngineContext) -> list[AdaptationProposalCandidate]:
        if not context.signals or not self._candidate_bundles:
            return []

        utilities = [signal.utility for signal in context.signals if signal.utility is not None]
        average_utility = sum(utilities) / len(utilities) if utilities else 0.0
        if average_utility >= self._utility_threshold:
            return []

        selected_bundle = self._candidate_bundles[0]
        signal_id = context.signals[-1].signal_id
        envelope = AdaptiveLoopEnvelope(
            loop_id=f"skill-selection-{context.task_class}",
            kind=AdaptiveLoopKind.ROUTING_TUNING,
            max_iterations=3,
            max_delta_percent=self._max_delta_percent,
            authority=AdaptiveAuthorityLevel.RECOMMEND,
            requires_human_approval=False,
            cooldown_seconds=900,
        )
        proposal = AdaptiveLoopProposal(
            envelope=envelope,
            proposed_change_summary=(
                f"Recommend skill bundle '{selected_bundle}' for task class '{context.task_class}'"
            ),
            evaluation_signal_id=signal_id,
        )
        draft = ProfileVersionDraft(
            version_id=f"draft-skill-{context.task_class}",
            artifact_type=ProfileArtifactType.ORCHESTRATION,
            artifact_payload={
                "skill_bundle_id": selected_bundle,
                "candidate_bundles": list(self._candidate_bundles),
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
