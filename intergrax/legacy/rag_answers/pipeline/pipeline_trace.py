# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import time
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


class StepTimer:

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def stop_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000