# © Artur Czarnecki. All rights reserved.

"""Default context fragment ranker (CE-4.4)."""

from __future__ import annotations

from intergrax.context.contracts import ContextAssemblyRequest, ContextFragment, ContextFragmentSource, replace_context_fragment
from intergrax.context.quality import ContextChunkSignal, evaluate_context_engineering

STEP_KIND_SOURCE_BOOSTS: dict[str, frozenset[ContextFragmentSource]] = {
    "tool_call": frozenset({ContextFragmentSource.TOOL_OUTPUT}),
    "retrieve": frozenset({ContextFragmentSource.RAG, ContextFragmentSource.WEBSEARCH}),
    "plan": frozenset({ContextFragmentSource.GRAPH_PRIOR, ContextFragmentSource.SHARED_CONTEXT}),
    "graph_node": frozenset({ContextFragmentSource.GRAPH_PRIOR, ContextFragmentSource.TASK_MESSAGE}),
}

_BOOST_DELTA = 0.15
_QUALITY_BYPASS_SOURCES = frozenset({ContextFragmentSource.SESSION_HISTORY})


class DefaultContextRanker:
    """Boosts fragments whose source matches the active ``step_kind`` (CE-4.4)."""

    ranker_id = "default"

    def rank(
        self,
        fragments: list[ContextFragment],
        request: ContextAssemblyRequest,
    ) -> list[ContextFragment]:
        ranked, _ = self.rank_with_exclusions(fragments, request)
        return ranked

    def rank_with_exclusions(
        self,
        fragments: list[ContextFragment],
        request: ContextAssemblyRequest,
    ) -> tuple[list[ContextFragment], list[tuple[ContextFragment, str]]]:
        session_fragments = [fragment for fragment in fragments if fragment.source is ContextFragmentSource.SESSION_HISTORY]
        non_session = [fragment for fragment in fragments if fragment.source is not ContextFragmentSource.SESSION_HISTORY]
        if not non_session and not request.step_kind:
            return self._partition_quality_gate(session_fragments)
        ranked_non_session, excluded = self._rank_non_session(non_session, request)
        included, quality_excluded = self._partition_quality_gate(ranked_non_session + session_fragments)
        return included, excluded + quality_excluded

    def _rank_non_session(
        self,
        fragments: list[ContextFragment],
        request: ContextAssemblyRequest,
    ) -> tuple[list[ContextFragment], list[tuple[ContextFragment, str]]]:
        if not fragments:
            return [], []
        if not request.step_kind:
            return (
                sorted(
                    fragments,
                    key=lambda item: (-item.normalized_relevance_score, item.fragment_id),
                ),
                [],
            )

        boosted_sources = STEP_KIND_SOURCE_BOOSTS.get(request.step_kind, frozenset())
        if not boosted_sources:
            return (
                sorted(
                    fragments,
                    key=lambda item: (-item.normalized_relevance_score, item.fragment_id),
                ),
                [],
            )

        ranked: list[ContextFragment] = []
        for fragment in fragments:
            if fragment.source in boosted_sources:
                boosted_score = min(1.0, fragment.normalized_relevance_score + _BOOST_DELTA)
                ranked.append(
                    replace_context_fragment(
                        fragment,
                        relevance_score=boosted_score,
                        normalized_relevance_score=boosted_score,
                    )
                )
            else:
                ranked.append(fragment)
        return (
            sorted(
                ranked,
                key=lambda item: (-item.normalized_relevance_score, item.fragment_id),
            ),
            [],
        )

    def _partition_quality_gate(
        self,
        fragments: list[ContextFragment],
    ) -> tuple[list[ContextFragment], list[tuple[ContextFragment, str]]]:
        if not fragments:
            return fragments, []
        signals = [
            ContextChunkSignal(
                chunk_id=fragment.fragment_id,
                content_hash=(
                    f"{fragment.source_id}:{fragment.content_hash}"
                    if fragment.source is ContextFragmentSource.SESSION_HISTORY
                    else fragment.content_hash
                ),
                relevance_score=fragment.normalized_relevance_score,
                freshness_score=fragment.freshness_score,
                confidence_score=fragment.confidence_score,
            )
            for fragment in fragments
        ]
        report = evaluate_context_engineering(chunks=signals)
        passed_ids = {record.chunk_id for record in report.records if record.passed}
        included = [
            fragment
            for fragment in fragments
            if fragment.fragment_id in passed_ids
            or fragment.source in _QUALITY_BYPASS_SOURCES
        ]
        excluded = [
            (fragment, "quality_threshold")
            for fragment in fragments
            if fragment.fragment_id not in passed_ids
            and fragment.source not in _QUALITY_BYPASS_SOURCES
        ]
        return included, excluded

    def _apply_quality_gate(self, fragments: list[ContextFragment]) -> list[ContextFragment]:
        included, _ = self._partition_quality_gate(fragments)
        return included
