# © Artur Czarnecki. All rights reserved.

"""Typed capability graph query contract (Phase CG-1)."""

from __future__ import annotations

from typing import Protocol

from intergrax.runtime.architecture.capability_graph import CapabilityGraph


class CapabilityGraphViewProtocol(Protocol):
    """Minimal surface for environment capability graph audits."""

    @property
    def graph(self) -> CapabilityGraph: ...

    def node_ids(self) -> tuple[str, ...]: ...

    def contains_node(self, node_id: str) -> bool: ...
