# © Artur Czarnecki. All rights reserved.

"""Runtime composition for LLM usage aggregation (EBH-2E-R6-R2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intergrax.llm_adapters.contracts.llm_usage_aggregation import LLMUsageAggregator
from intergrax.llm_adapters.tracking.llm_usage_track import LLMUsageTracker

if TYPE_CHECKING:
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState


def create_platform_default_llm_usage_tracker(run_id: str) -> LLMUsageAggregator:
    """Instantiate the platform default per-run usage aggregator."""
    return LLMUsageTracker(run_id=run_id)


def ensure_llm_usage_tracker_on_state(state: RuntimeState) -> None:
    """Ensure ``RuntimeState`` holds a usage aggregator via runtime composition."""
    if state.llm_usage_tracker is None:
        state.llm_usage_tracker = create_platform_default_llm_usage_tracker(state.run_id)


__all__ = [
    "create_platform_default_llm_usage_tracker",
    "ensure_llm_usage_tracker_on_state",
]
