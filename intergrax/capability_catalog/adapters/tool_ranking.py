# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool keyword-overlap ranking adapter (Stage 4)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from intergrax.capability_catalog.candidate import CapabilityDiscoveryCandidate
from intergrax.capability_catalog.ranked_candidate import RankedCapabilityCandidate
from intergrax.capability_catalog.ranking import identity_sort_key
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_catalog.ranking import (
    CapabilityRankingContext,
    CapabilityRankingEvidence,
    CapabilityRankingSignal,
)
from intergrax.tools.search.keyword_ranking import (
    ToolKeywordSearchDocument,
    score_tool_keyword_document,
    tokenize_tool_search_query,
)

KEYWORD_OVERLAP_TOOL_RANKER_ID: Final = "tool.keyword_overlap"


def _tool_keyword_search_document(
    candidate: CapabilityDiscoveryCandidate,
) -> ToolKeywordSearchDocument:
    return ToolKeywordSearchDocument(
        tool_id=candidate.identity.logical.logical_id,
        text_parts=tuple(
            part
            for part in (candidate.catalog_entry.display_label,)
            if part
        ),
    )


class KeywordOverlapToolCapabilityRanker:
    """Keyword overlap ranker using shared Tool-domain scoring primitive — ordering only."""

    @property
    def ranker_id(self) -> str:
        return KEYWORD_OVERLAP_TOOL_RANKER_ID

    def rank(
        self,
        candidates: tuple[CapabilityDiscoveryCandidate, ...],
        context: CapabilityRankingContext,
    ) -> tuple[RankedCapabilityCandidate, ...]:
        query_tokens = tokenize_tool_search_query(context.semantic_need or "")
        indexed = tuple((index, candidate) for index, candidate in enumerate(candidates))
        ordered = sorted(
            indexed,
            key=lambda pair: (
                -_tool_ranking_score(pair[1], query_tokens),
                identity_sort_key(pair[1]),
                pair[0],
            ),
        )
        return tuple(
            RankedCapabilityCandidate(
                candidate=candidate,
                evidence=CapabilityRankingEvidence(
                    ranker_id=self.ranker_id,
                    rank_position=position,
                    signal=CapabilityRankingSignal.KEYWORD_OVERLAP,
                    score=float(_tool_ranking_score(candidate, query_tokens)),
                    original_stage3_position=original_index + 1,
                ),
            )
            for position, (original_index, candidate) in enumerate(ordered, start=1)
        )


def _tool_ranking_score(
    candidate: CapabilityDiscoveryCandidate,
    query_tokens: Sequence[str],
) -> int:
    if candidate.identity.kind is not CapabilityKind.TOOL:
        return 0
    return score_tool_keyword_document(
        _tool_keyword_search_document(candidate),
        query_tokens,
    )
