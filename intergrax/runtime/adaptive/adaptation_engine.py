# © Artur Czarnecki. All rights reserved.

"""Adaptation engine facade for L4-R recommend wave (Phase W-ADAPT-2.7)."""

from __future__ import annotations

from intergrax.runtime.adaptive.adaptation_models import (
    AdaptationEngineContext,
    AdaptationEngineRunResult,
    AdaptationProposalCandidate,
)
from intergrax.runtime.adaptive.adaptation_sub_engine import AdaptationSubEngine
from intergrax.runtime.adaptive.bandit_state_store import BanditStateStore
from intergrax.runtime.adaptive.cost_anomaly_bridge import proposals_from_cost_anomalies
from intergrax.runtime.adaptive.proposal_builder import ProposalBuilder
from intergrax.runtime.adaptive.proposal_cooldown_store import ProposalCooldownStore
from intergrax.runtime.adaptive.proposal_store import ProposalStore


def _utility_reward(utility: float | None) -> float:
    if utility is None:
        return 0.5
    normalized = (utility + 1.0) / 2.0
    return max(0.0, min(1.0, normalized))


class AdaptationEngine:
    """Rank and gate proposal candidates from sub-engines (recommend-only)."""

    def __init__(
        self,
        *,
        sub_engines: list[AdaptationSubEngine],
        proposal_builder: ProposalBuilder,
        bandit_store: BanditStateStore,
        cooldown_store: ProposalCooldownStore,
        proposal_store: ProposalStore | None = None,
        default_arm_id: str = "rag_tier_default",
    ) -> None:
        self._sub_engines = sub_engines
        self._proposal_builder = proposal_builder
        self._bandit_store = bandit_store
        self._cooldown_store = cooldown_store
        self._proposal_store = proposal_store
        self._default_arm_id = default_arm_id

    def run(self, context: AdaptationEngineContext) -> AdaptationEngineRunResult:
        self._update_bandit_from_signals(context)
        candidates = self._collect_candidates(context)
        filtered, skipped = self._apply_cooldown(candidates)
        ranked = sorted(filtered, key=lambda item: item.rank_score, reverse=True)
        packages = [
            self._proposal_builder.build_package(candidate, context=context)
            for candidate in ranked
        ]
        for candidate in ranked:
            self._cooldown_store.mark_proposed(candidate.loop_id)

        result = AdaptationEngineRunResult(
            tenant_id=context.tenant_id,
            task_class=context.task_class,
            packages=packages,
            skipped_cooldown_loop_ids=skipped,
        )
        if self._proposal_store is not None:
            self._proposal_store.append_run(result)
        return result

    def _update_bandit_from_signals(self, context: AdaptationEngineContext) -> None:
        for signal in context.signals:
            reward = _utility_reward(signal.utility)
            self._bandit_store.record_reward(
                tenant_id=context.tenant_id,
                task_class=context.task_class,
                arm_id=self._default_arm_id,
                reward=reward,
            )

    def _collect_candidates(
        self,
        context: AdaptationEngineContext,
    ) -> list[AdaptationProposalCandidate]:
        candidates: list[AdaptationProposalCandidate] = []
        for engine in self._sub_engines:
            candidates.extend(engine.propose(context))
        candidates.extend(proposals_from_cost_anomalies(context))
        return candidates

    def _apply_cooldown(
        self,
        candidates: list[AdaptationProposalCandidate],
    ) -> tuple[list[AdaptationProposalCandidate], list[str]]:
        accepted: list[AdaptationProposalCandidate] = []
        skipped: list[str] = []
        for candidate in candidates:
            if self._cooldown_store.is_on_cooldown(
                candidate.loop_id,
                cooldown_seconds=candidate.cooldown_seconds,
            ):
                skipped.append(candidate.loop_id)
                continue
            accepted.append(candidate)
        return accepted, skipped
