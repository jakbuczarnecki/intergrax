# © Artur Czarnecki. All rights reserved.

"""Hybrid retrieval reference path contracts (Phase V-KG.2)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RetrievalChannel(str, Enum):
    VECTOR = "vector"
    KEYWORD = "keyword"
    GRAPH = "graph"


class ChannelRetrievalHit(BaseModel):
    channel: RetrievalChannel
    document_id: str
    score: float


class HybridRetrievalRequest(BaseModel):
    query_id: str
    vector_hits: list[ChannelRetrievalHit] = Field(default_factory=list)
    keyword_hits: list[ChannelRetrievalHit] = Field(default_factory=list)
    graph_hits: list[ChannelRetrievalHit] = Field(default_factory=list)
    top_k: int = 5


class HybridRetrievalResult(BaseModel):
    query_id: str
    merged_document_ids: list[str] = Field(default_factory=list)
    channel_contributions: dict[str, list[str]] = Field(default_factory=dict)


class HybridRetrievalReport(BaseModel):
    schema_version: str = "1.0.0"
    results: list[HybridRetrievalResult] = Field(default_factory=list)


def execute_hybrid_retrieval(request: HybridRetrievalRequest) -> HybridRetrievalResult:
    ranked: dict[str, float] = {}
    contributions: dict[str, list[str]] = {
        RetrievalChannel.VECTOR.value: [],
        RetrievalChannel.KEYWORD.value: [],
        RetrievalChannel.GRAPH.value: [],
    }
    for hits, channel in (
        (request.vector_hits, RetrievalChannel.VECTOR),
        (request.keyword_hits, RetrievalChannel.KEYWORD),
        (request.graph_hits, RetrievalChannel.GRAPH),
    ):
        for hit in hits:
            ranked[hit.document_id] = ranked.get(hit.document_id, 0.0) + hit.score
            contributions[channel.value].append(hit.document_id)
    merged = [
        document_id
        for document_id, _score in sorted(
            ranked.items(),
            key=lambda item: item[1],
            reverse=True,
        )[: request.top_k]
    ]
    return HybridRetrievalResult(
        query_id=request.query_id,
        merged_document_ids=merged,
        channel_contributions=contributions,
    )
