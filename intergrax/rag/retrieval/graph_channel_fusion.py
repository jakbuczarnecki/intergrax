# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Graph + vector channel fusion for GraphRagRetriever (M-RAG.43) — Tier-0 only."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class GraphChannelHit:
    document_id: str
    score: float
    channel: str


@dataclass(frozen=True)
class GraphChannelFusionResult:
    merged_document_ids: List[str]
    channel_contributions: Dict[str, List[str]] = field(default_factory=dict)


def fuse_graph_channels(
    *,
    vector_hits: List[GraphChannelHit],
    graph_hits: List[GraphChannelHit],
    top_k: int,
) -> GraphChannelFusionResult:
    """Score-sum merge aligned with ``execute_hybrid_retrieval`` reference semantics."""
    ranked: Dict[str, float] = {}
    contributions: Dict[str, List[str]] = {"vector": [], "graph": []}
    for hit in vector_hits:
        ranked[hit.document_id] = ranked.get(hit.document_id, 0.0) + hit.score
        contributions["vector"].append(hit.document_id)
    for hit in graph_hits:
        ranked[hit.document_id] = ranked.get(hit.document_id, 0.0) + hit.score
        contributions["graph"].append(hit.document_id)
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
