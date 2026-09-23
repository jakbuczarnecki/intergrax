# © Artur Czarnecki. All rights reserved.

"""Live routing profile composition source (internal platform contract)."""

from __future__ import annotations

from typing import Protocol

from intergrax.llm_adapters.contracts.routing_profile import LLMRoutingProfile


class RoutingProfileSource(Protocol):
    """Exposes the current routing profile for live re-evaluation (read-only)."""

    @property
    def llm_routing_profile(self) -> LLMRoutingProfile | None:
        ...
