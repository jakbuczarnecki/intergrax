# © Artur Czarnecki. All rights reserved.

"""Orchestration ceiling patch helper (ECP-PROD.5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class OrchestrationCeilingPatcher(Protocol):
    def raise_ceiling(self, *, delta: int) -> int: ...


@dataclass
class BoundedOrchestrationCeilingPatcher:
    """Raise max_inflight_nodes within a bounded percent cap."""

    max_inflight_nodes: int
    max_raise_percent: int = 15

    def raise_ceiling(self, *, delta: int) -> int:
        if delta <= 0:
            return self.max_inflight_nodes
        absolute_cap = max(
            self.max_inflight_nodes,
            int(self.max_inflight_nodes * (1 + self.max_raise_percent / 100)),
        )
        self.max_inflight_nodes = min(absolute_cap, self.max_inflight_nodes + delta)
        return self.max_inflight_nodes
