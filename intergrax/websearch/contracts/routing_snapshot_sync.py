# © Artur Czarnecki. All rights reserved.

"""WebSearch-owned port for live LLM routing snapshot refresh before grounding LLM calls."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class WebSearchLlmRoutingSnapshotSync(Protocol):
    """Refresh routing snapshot immediately before a WebSearch-layer LLM invocation."""

    def sync_before_llm_call(self, *, run_id: str | None) -> None: ...


__all__ = ["WebSearchLlmRoutingSnapshotSync"]
