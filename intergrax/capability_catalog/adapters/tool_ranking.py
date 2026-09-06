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

KEYWORD_OVERLAP_TOOL_RANKER_ID: Final = "tool.keyword_overlap"


def _query_tokens(query: str) -> tuple[str, ...]:
    return tuple(token for token in query.lower().split() if len(token) > 2)


def _keyword_overlap_score(
    candidate: CapabilityDiscoveryCandidate,
    query_tokens: Sequence[str],
) -> int:
    if not query_tokens:
        return 0
    haystack = " ".join(
        (
            candidate.identity.logical.logical_id,
            candidate.catalog_entry.display_label,
        ),
    ).lower()
    return sum(1 for token in query_tokens if token in haystack)


class KeywordOverlapToolCapabilityRanker:
    """Keyword overlap ranker adapted from TOOL-ENG-5 retrieval scoring — ordering only."""

    @property
    def ranker_id(self) -> str:
        return KEYWORD_OVERLAP_TOOL_RANKER_ID

    def rank(
        self,
        candidates: tuple[CapabilityDiscoveryCandidate, ...],
        context: CapabilityRankingContext,
    ) -> tuple[RankedCapabilityCandidate, ...]:
        query_tokens = _query_tokens(context.semantic_need or "")
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
    return _keyword_overlap_score(candidate, query_tokens)
