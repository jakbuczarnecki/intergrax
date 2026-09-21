# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Host-supplied canonical catalog tool invocation binding (UCA-6C-R5)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.runtime.nexus.tools.tool_invoker_protocol import ToolInvokerProtocol


@runtime_checkable
class CatalogToolInvocationPort(Protocol):
    """Canonical ToolRuntime invoker plus trusted runtime state for one execution scope."""

    @property
    def tool_invoker(self) -> ToolInvokerProtocol: ...

    def runtime_state_for_invocation(self) -> object: ...

    @property
    def caller_agent_id(self) -> str: ...


@dataclass(frozen=True, slots=True)
class CatalogToolInvocationBinding:
    """Default composition-owned binding; state must come from host/runtime wiring."""

    tool_invoker: ToolInvokerProtocol
    state_supplier: Callable[[], object]
    caller_agent_id: str

    def runtime_state_for_invocation(self) -> object:
        return self.state_supplier()


__all__ = [
    "CatalogToolInvocationBinding",
    "CatalogToolInvocationPort",
]
