# © Artur Czarnecki. All rights reserved.

"""Canonical LLM usage aggregation port (EBH-2E-R6-R2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.llm_adapters.contracts.llm_usage_stats import LLMUsageTrackable
from intergrax.llm_adapters.contracts.llm_usage_report import LLMUsageReport


@runtime_checkable
class LLMUsageAggregator(Protocol):
    """Per-run usage collection across registered adapters."""

    def register_adapter(
        self,
        trackable: LLMUsageTrackable,
        label: str | None = None,
    ) -> None:
        """Register a usage-trackable adapter for this run."""

    def build_report(self) -> LLMUsageReport:
        """Build the aggregated usage report for this run."""


__all__ = ["LLMUsageAggregator"]
