# © Artur Czarnecki. All rights reserved.

"""Canonical LLM run statistics and usage-source contracts (EBH-2E-R6-R2-R2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider


@dataclass
class LLMRunStats:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    duration_ms: int = 0
    errors: int = 0


@runtime_checkable
class LLMRunStatsReader(Protocol):
    """Per-adapter run-level usage snapshot access (platform contract)."""

    def get_run_stats(self, run_id: Optional[str] = None) -> LLMRunStats | None:
        ...


@runtime_checkable
class LLMUsageTrackable(Protocol):
    """Adapter identity + usage stats source for usage aggregation registration."""

    provider: LLMProvider | str
    model: str
    usage: LLMRunStatsReader


__all__ = [
    "LLMRunStats",
    "LLMRunStatsReader",
    "LLMUsageTrackable",
]
