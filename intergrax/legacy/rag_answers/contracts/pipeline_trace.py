# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineTrace:
    retrieval_latency_ms: float | None = None
    rerank_latency_ms: float | None = None
    context_latency_ms: float | None = None
    prompt_latency_ms: float | None = None
    llm_latency_ms: float | None = None
    retrieved_candidates: int | None = None
    reranked_candidates: int | None = None


__all__ = ["PipelineTrace"]
