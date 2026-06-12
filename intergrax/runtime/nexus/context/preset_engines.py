# © Artur Czarnecki. All rights reserved.

"""Shipped context engine presets (CE-PRESET-ENG)."""

from __future__ import annotations

from intergrax.context.contracts import ContextFragment
from intergrax.context.quality import ContextChunkSignal, ContextQualityThresholds, evaluate_context_engineering
from intergrax.context.ranker import DefaultContextRanker
from intergrax.context.registry import ContextPluginRegistry
from intergrax.runtime.nexus.context.context_compiler import ContextCompiler
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine
from intergrax.runtime.nexus.context.context_validator import DefaultContextValidator

_REGULATED_THRESHOLDS = ContextQualityThresholds(
    min_relevance=0.75,
    min_freshness=0.60,
    min_confidence=0.85,
    min_composite_score=0.80,
)

_EXPLORE_CHILD_MAX_FRAGMENTS = 4


class RegulatedMinimalContextRanker(DefaultContextRanker):
    """High-confidence gate for regulated_minimal preset."""

    ranker_id = "regulated_minimal"

    def _partition_quality_gate(
        self,
        fragments: list[ContextFragment],
    ) -> tuple[list[ContextFragment], list[tuple[ContextFragment, str]]]:
        if not fragments:
            return fragments, []
        signals = [
            ContextChunkSignal(
                chunk_id=fragment.fragment_id,
                content_hash=fragment.content_hash,
                relevance_score=fragment.relevance_score,
                freshness_score=fragment.freshness_score,
                confidence_score=fragment.confidence_score,
            )
            for fragment in fragments
        ]
        report = evaluate_context_engineering(chunks=signals, thresholds=_REGULATED_THRESHOLDS)
        passed_ids = {record.chunk_id for record in report.records if record.passed}
        included = [fragment for fragment in fragments if fragment.fragment_id in passed_ids]
        excluded = [
            (fragment, "regulated_quality_threshold")
            for fragment in fragments
            if fragment.fragment_id not in passed_ids
        ]
        return included, excluded


class ExploreChildContextRanker(DefaultContextRanker):
    """Tight fragment cap for delegation explore children."""

    ranker_id = "explore_child"

    def rank_with_exclusions(
        self,
        fragments: list[ContextFragment],
        request,
    ) -> tuple[list[ContextFragment], list[tuple[ContextFragment, str]]]:
        ranked, excluded = super().rank_with_exclusions(fragments, request)
        if len(ranked) <= _EXPLORE_CHILD_MAX_FRAGMENTS:
            return ranked, excluded
        kept = ranked[:_EXPLORE_CHILD_MAX_FRAGMENTS]
        dropped = [(fragment, "explore_child_cap") for fragment in ranked[_EXPLORE_CHILD_MAX_FRAGMENTS:]]
        return kept, excluded + dropped


class RegulatedMinimalContextEngine(DefaultNexusContextEngine):
    def __init__(
        self,
        *,
        registry: ContextPluginRegistry | None = None,
        compiler: ContextCompiler | None = None,
        validator: DefaultContextValidator | None = None,
    ) -> None:
        super().__init__(
            engine_id="regulated_minimal",
            registry=registry,
            compiler=compiler,
            validator=validator,
            ranker=RegulatedMinimalContextRanker(),
        )


class ExploreChildContextEngine(DefaultNexusContextEngine):
    def __init__(
        self,
        *,
        registry: ContextPluginRegistry | None = None,
        compiler: ContextCompiler | None = None,
        validator: DefaultContextValidator | None = None,
    ) -> None:
        super().__init__(
            engine_id="explore_child",
            registry=registry,
            compiler=compiler,
            validator=validator,
            ranker=ExploreChildContextRanker(),
        )
