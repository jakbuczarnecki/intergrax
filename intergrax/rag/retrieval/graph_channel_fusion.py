# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Graph + vector + keyword channel fusion for GraphRagRetriever (M-RAG.43, M-RAG.53) — Tier-0 only."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class GraphChannelHit:
    document_id: str
    score: float
    channel: str


@dataclass(frozen=True)
class GraphChannelFusionResult:
    merged_document_ids: List[str]
    channel_contributions: Dict[str, List[str]] = field(default_factory=dict)


def lexical_score(query_text: str, document_text: str) -> float:
    """Token overlap score aligned with ``HybridRetriever`` lexical channel."""
    query_tokens = re.findall(r"\w+", (query_text or "").lower())
    doc_tokens = re.findall(r"\w+", (document_text or "").lower())
    if not query_tokens or not doc_tokens:
        return 0.0
    matches = sum(1 for token in query_tokens if token in doc_tokens)
    return matches / len(query_tokens)


def build_keyword_hits(
    *,
    query_text: str,
    candidates: Sequence[tuple[str, str]],
) -> List[GraphChannelHit]:
    """Score document text candidates for the keyword retrieval channel."""
    hits: List[GraphChannelHit] = []
    for document_id, document_text in candidates:
        score = lexical_score(query_text, document_text)
        if score <= 0.0:
            continue
        hits.append(
            GraphChannelHit(
                document_id=document_id,
                score=score,
                channel="keyword",
            )
        )
    return hits


def fuse_graph_channels(
    *,
    vector_hits: List[GraphChannelHit],
    graph_hits: List[GraphChannelHit],
    keyword_hits: List[GraphChannelHit] | None = None,
    top_k: int,
) -> GraphChannelFusionResult:
    """Score-sum merge aligned with ``execute_hybrid_retrieval`` reference semantics."""
    keyword_hits = keyword_hits or []
    ranked: Dict[str, float] = {}
    contributions: Dict[str, List[str]] = {"vector": [], "keyword": [], "graph": []}
    for hits, channel in (
        (vector_hits, "vector"),
        (keyword_hits, "keyword"),
        (graph_hits, "graph"),
    ):
        for hit in hits:
            ranked[hit.document_id] = ranked.get(hit.document_id, 0.0) + hit.score
            contributions[channel].append(hit.document_id)
    merged = [
        document_id
        for document_id, _score in sorted(
            ranked.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
    ]
    return GraphChannelFusionResult(
        merged_document_ids=merged,
        channel_contributions=contributions,
    )
