# © Artur Czarnecki. All rights reserved.

"""Token metering helpers for routing context (M-LLM-X.12.1)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter


@runtime_checkable
class _InnerAdapterCarrier(Protocol):
    @property
    def inner_adapter(self) -> LLMAdapter: ...


def resolve_metering_adapter(adapter: LLMAdapter | None) -> LLMAdapter | None:
    """Return the adapter that accumulates LLM usage (unwrap evaluating wrappers)."""
    if adapter is None:
        return None
    if isinstance(adapter, _InnerAdapterCarrier):
        return adapter.inner_adapter
    return adapter


def tokens_used_from_adapter(adapter: LLMAdapter | None, *, run_id: str | None = None) -> int:
    """Read aggregated token usage from the metering adapter for a run."""
    core = resolve_metering_adapter(adapter)
    if core is None:
        return 0
    stats = core.usage.get_run_stats(run_id)
    return int(stats.total_tokens)
