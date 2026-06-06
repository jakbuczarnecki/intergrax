# © Artur Czarnecki. All rights reserved.

"""Routing tuning sub-engine (Phase W-ADAPT-2.2)."""

from __future__ import annotations

from intergrax.runtime.adaptive.adaptation_models import (
    AdaptationEngineContext,
    AdaptationProposalCandidate,
)
from intergrax.runtime.adaptive.bandit_state_store import BanditStateStore
from intergrax.runtime.adaptive.contracts import ProfileArtifactType, ProfileVersionDraft
from intergrax.runtime.architecture.adaptive_governance import (
    AdaptiveAuthorityLevel,
    AdaptiveLoopEnvelope,
    AdaptiveLoopKind,
    AdaptiveLoopProposal,
)

_DEFAULT_ARMS: tuple[str, ...] = ("rag_tier_default", "rag_tier_deep", "llm_route_balanced")


class RoutingTuningEngine:
    """Contextual bandit routing proposals for LLM/RAG tier arms."""

    def __init__(
        self,
        bandit_store: BanditStateStore,
        *,
        max_delta_percent: float = 10.0,
        utility_threshold: float = 0.45,
    ) -> None:
        self._bandit_store = bandit_store
        self._max_delta_percent = max_delta_percent
        self._utility_threshold = utility_threshold

    @property
    def engine_id(self) -> str:
        return "routing_tuning"

    def propose(self, context: AdaptationEngineContext) -> list[AdaptationProposalCandidate]:
        if not context.signals:
            return []

        utilities = [signal.utility for signal in context.signals if signal.utility is not None]
        average_utility = sum(utilities) / len(utilities) if utilities else 0.0
        if average_utility >= self._utility_threshold:
            return []

        arm_scores = {
            arm_id: self._bandit_store.sample_arm_score(
                tenant_id=context.tenant_id,
                task_class=context.task_class,
                arm_id=arm_id,
            )
            for arm_id in _DEFAULT_ARMS
        }
        selected_arm = max(arm_scores, key=arm_scores.get)
        signal_id = context.signals[-1].signal_id

        envelope = AdaptiveLoopEnvelope(
            loop_id=f"routing-tuning-{context.task_class}",
            kind=AdaptiveLoopKind.ROUTING_TUNING,
            max_iterations=5,
            max_delta_percent=self._max_delta_percent,
            authority=AdaptiveAuthorityLevel.RECOMMEND,
            requires_human_approval=False,
            cooldown_seconds=3600,
        )
        proposal = AdaptiveLoopProposal(
            envelope=envelope,
            proposed_change_summary=(
                f"Recommend routing shift toward arm '{selected_arm}' "
                f"for task_class '{context.task_class}' (avg utility {average_utility:.2f})"
            ),
            evaluation_signal_id=signal_id,
        )
        draft = ProfileVersionDraft(
            version_id=f"draft-routing-{context.task_class}-{selected_arm}",
            artifact_type=ProfileArtifactType.RAG,
            artifact_payload={"selected_arm": selected_arm, "max_delta_percent": self._max_delta_percent},
            created_by=self.engine_id,
        )
        return [
            AdaptationProposalCandidate(
                loop_id=envelope.loop_id,
                source_engine=self.engine_id,
                proposal=proposal,
                profile_draft=draft,
                rank_score=1.0 - average_utility,
                cooldown_seconds=envelope.cooldown_seconds,
            )
        ]
