# © Artur Czarnecki. All rights reserved.

"""Default context fragment ranker (CE-4.4)."""

from __future__ import annotations

from intergrax.context.contracts import ContextAssemblyRequest, ContextFragment, ContextFragmentSource
from intergrax.context.quality import ContextChunkSignal, evaluate_context_engineering

STEP_KIND_SOURCE_BOOSTS: dict[str, frozenset[ContextFragmentSource]] = {
    "tool_call": frozenset({ContextFragmentSource.TOOL_OUTPUT}),
    "retrieve": frozenset({ContextFragmentSource.RAG, ContextFragmentSource.WEBSEARCH}),
    "plan": frozenset({ContextFragmentSource.GRAPH_PRIOR, ContextFragmentSource.SHARED_CONTEXT}),
    "graph_node": frozenset({ContextFragmentSource.GRAPH_PRIOR, ContextFragmentSource.TASK_MESSAGE}),
}

_BOOST_DELTA = 0.15


class DefaultContextRanker:
    """Boosts fragments whose source matches the active ``step_kind`` (CE-4.4)."""

    ranker_id = "default"

    def rank(
        self,
        fragments: list[ContextFragment],
        request: ContextAssemblyRequest,
    ) -> list[ContextFragment]:
        if not fragments or not request.step_kind:
            return self._apply_quality_gate(
                sorted(fragments, key=lambda item: item.relevance_score, reverse=True)
            )

        boosted_sources = STEP_KIND_SOURCE_BOOSTS.get(request.step_kind, frozenset())
        if not boosted_sources:
            return self._apply_quality_gate(
                sorted(fragments, key=lambda item: item.relevance_score, reverse=True)
            )

        ranked: list[ContextFragment] = []
        for fragment in fragments:
            if fragment.source in boosted_sources:
                boosted_score = min(1.0, fragment.relevance_score + _BOOST_DELTA)
                ranked.append(
                    ContextFragment(
                        fragment_id=fragment.fragment_id,
                        source=fragment.source,
                        source_id=fragment.source_id,
                        content=fragment.content,
                        token_estimate=fragment.token_estimate,
                        relevance_score=boosted_score,
                        freshness_score=fragment.freshness_score,
                        confidence_score=fragment.confidence_score,
                        mandatory=fragment.mandatory,
                        metadata=dict(fragment.metadata),
                        content_hash=fragment.content_hash,
                    )
                )
            else:
                ranked.append(fragment)
        return self._apply_quality_gate(
            sorted(ranked, key=lambda item: item.relevance_score, reverse=True)
        )

    def _apply_quality_gate(self, fragments: list[ContextFragment]) -> list[ContextFragment]:
        if not fragments:
            return fragments
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
        report = evaluate_context_engineering(chunks=signals)
        passed_ids = {record.chunk_id for record in report.records if record.passed}
        return [fragment for fragment in fragments if fragment.fragment_id in passed_ids]
