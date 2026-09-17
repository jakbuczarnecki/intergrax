# © Artur Czarnecki. All rights reserved.

"""Cross-source policy pipeline orchestrator (MEM-XINT-5)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextFragment,
    ContextNormalizationInput,
    ContextPolicyDecision,
    ContextPolicyPipelineResult,
    ContextPolicyReasonCode,
    ContextPolicyStage,
)
from intergrax.context.policy.scope_isolation import isolate_assembly_scope
from intergrax.context.policy.budget_allocator import DefaultContextBudgetAllocator
from intergrax.context.policy.canonicalization import canonicalize_fragment_for_policy
from intergrax.context.policy.conflict_resolver import DefaultContextConflictResolver
from intergrax.context.policy.config import ContextPolicyPipelineConfig
from intergrax.context.policy.exact_dedup import exact_dedup_fragments
from intergrax.context.policy.score_normalizer import DefaultContextScoreNormalizer
from intergrax.context.policy.semantic_dedup import DefaultContextSemanticDeduper
from intergrax.context.ranker import DefaultContextRanker
from intergrax.context.protocols import (
    ContextBudgetAllocator,
    ContextConflictResolver,
    ContextRanker,
    ContextScoreNormalizer,
    ContextSemanticDeduper,
)


def _fragment_ids(fragments: list[ContextFragment]) -> tuple[str, ...]:
    return tuple(fragment.fragment_id for fragment in fragments)


def _decision(
    *,
    stage: ContextPolicyStage,
    strategy_id: str,
    before: list[ContextFragment],
    after: list[ContextFragment],
    reason_code: ContextPolicyReasonCode,
    detail: str = "",
) -> ContextPolicyDecision:
    return ContextPolicyDecision(
        stage=stage,
        strategy_id=strategy_id,
        input_fragment_ids=_fragment_ids(before),
        output_fragment_ids=_fragment_ids(after),
        reason_code=reason_code,
        detail=detail,
    )


@dataclass(frozen=True, slots=True)
class ContextPolicyStrategies:
    score_normalizer: ContextScoreNormalizer
    semantic_deduper: ContextSemanticDeduper
    conflict_resolver: ContextConflictResolver
    ranker: ContextRanker
    budget_allocator: ContextBudgetAllocator


def default_context_policy_strategies() -> ContextPolicyStrategies:
    return ContextPolicyStrategies(
        score_normalizer=DefaultContextScoreNormalizer(),
        semantic_deduper=DefaultContextSemanticDeduper(),
        conflict_resolver=DefaultContextConflictResolver(),
        ranker=DefaultContextRanker(),
        budget_allocator=DefaultContextBudgetAllocator(),
    )


class ContextCrossSourcePolicyPipeline:
    """Deterministic CE policy stages between collect and compile."""

    @property
    def pipeline_id(self) -> str:
        return "intergrax.context.cross_source_policy.v1"

    def __init__(
        self,
        *,
        strategies: ContextPolicyStrategies | None = None,
        config: ContextPolicyPipelineConfig | None = None,
    ) -> None:
        self._strategies = strategies or default_context_policy_strategies()
        self._config = config or ContextPolicyPipelineConfig()

    def execute(
        self,
        fragments: list[ContextFragment],
        request: ContextAssemblyRequest,
        *,
        strategies: ContextPolicyStrategies | None = None,
    ) -> ContextPolicyPipelineResult:
        active_strategies = strategies or self._strategies
        decisions: list[ContextPolicyDecision] = []
        excluded: list[tuple[ContextFragment, str]] = []
        working = list(fragments)

        before = list(working)
        working, scope_excluded = isolate_assembly_scope(working, request)
        excluded.extend(scope_excluded)
        decisions.append(
            _decision(
                stage=ContextPolicyStage.SCOPE_ISOLATION,
                strategy_id="platform.scope_isolation.v1",
                before=before,
                after=working,
                reason_code=ContextPolicyReasonCode.SCOPE_INCOMPATIBLE,
            ),
        )

        before = list(working)
        working = [canonicalize_fragment_for_policy(fragment) for fragment in working]
        decisions.append(
            _decision(
                stage=ContextPolicyStage.CANONICALIZE,
                strategy_id="platform.canonicalize.v1",
                before=before,
                after=working,
                reason_code=ContextPolicyReasonCode.CONFLICT_RESOLVED,
                detail="canonical_content",
            ),
        )

        before = list(working)
        working, exact_dropped, _exact_audit = exact_dedup_fragments(working)
        excluded.extend(exact_dropped)
        decisions.append(
            _decision(
                stage=ContextPolicyStage.EXACT_DEDUP,
                strategy_id="platform.exact_dedup.v1",
                before=before,
                after=working,
                reason_code=ContextPolicyReasonCode.EXACT_DUPLICATE_CONTENT,
            ),
        )

        before = list(working)
        normalized: list[ContextFragment] = []
        for fragment in working:
            normalized.append(
                active_strategies.score_normalizer.normalize(
                    ContextNormalizationInput(fragment=fragment, request=request),
                ),
            )
        working = normalized
        decisions.append(
            _decision(
                stage=ContextPolicyStage.NORMALIZE,
                strategy_id=active_strategies.score_normalizer.strategy_id,
                before=before,
                after=working,
                reason_code=ContextPolicyReasonCode.CONFLICT_RESOLVED,
                detail="score_normalization",
            ),
        )

        before = list(working)
        working, semantic_decisions = active_strategies.semantic_deduper.deduplicate(working, request)
        if semantic_decisions:
            suppressed_ids = {
                suppressed
                for decision in semantic_decisions
                for suppressed in decision.suppressed_fragment_ids
            }
            id_to_fragment = {fragment.fragment_id: fragment for fragment in before}
            for suppressed_id in sorted(suppressed_ids):
                fragment = id_to_fragment.get(suppressed_id)
                if fragment is not None:
                    excluded.append((fragment, ContextPolicyReasonCode.SEMANTIC_DUPLICATE.value))
        decisions.append(
            _decision(
                stage=ContextPolicyStage.SEMANTIC_DEDUP,
                strategy_id=active_strategies.semantic_deduper.strategy_id,
                before=before,
                after=working,
                reason_code=ContextPolicyReasonCode.SEMANTIC_DUPLICATE,
            ),
        )

        before = list(working)
        working, conflict_decisions = active_strategies.conflict_resolver.resolve(working, request)
        decisions.append(
            _decision(
                stage=ContextPolicyStage.CONFLICT,
                strategy_id=active_strategies.conflict_resolver.strategy_id,
                before=before,
                after=working,
                reason_code=ContextPolicyReasonCode.CONFLICT_RESOLVED,
            ),
        )

        before = list(working)
        ranked, quality_excluded = active_strategies.ranker.rank_with_exclusions(working, request)
        excluded.extend(quality_excluded)
        working = ranked
        decisions.append(
            _decision(
                stage=ContextPolicyStage.RANK,
                strategy_id=active_strategies.ranker.ranker_id,
                before=before,
                after=working,
                reason_code=ContextPolicyReasonCode.QUALITY_THRESHOLD,
            ),
        )

        before = list(working)
        allocation = active_strategies.budget_allocator.allocate(
            working,
            request.budget_policy.max_tokens_estimate,
            request,
        )
        working = list(allocation.included)
        excluded.extend(allocation.excluded)
        decisions.append(
            _decision(
                stage=ContextPolicyStage.BUDGET,
                strategy_id=active_strategies.budget_allocator.strategy_id,
                before=before,
                after=working,
                reason_code=ContextPolicyReasonCode.BUDGET_EXCLUDED,
            ),
        )

        return ContextPolicyPipelineResult(
            fragments=tuple(working),
            excluded=tuple(excluded),
            decisions=tuple(decisions),
            semantic_dedup_decisions=semantic_decisions,
            conflict_decisions=conflict_decisions,
        )
