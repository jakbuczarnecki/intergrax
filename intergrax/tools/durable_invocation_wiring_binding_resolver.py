# © Artur Czarnecki. All rights reserved.

"""Durable runtime binding resolution for tool invocation wiring (UCA-6C-R6-R3)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.tools.invocation_wiring import ToolInvocationWiringResolver


class DurableToolInvocationWiringBindingResolutionError(Exception):
    """Fail-closed binding materialization failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@runtime_checkable
class DurableToolInvocationWiringBindingResolver(Protocol):
    """Provider-neutral resolver from durable references to invocation wiring."""

    def resolve_fixed_sandbox_session_wiring(
        self,
        *,
        sandbox_session_id: str,
        tenant_id: str,
        task_id: str,
    ) -> ToolInvocationWiringResolver: ...


__all__ = [
    "DurableToolInvocationWiringBindingResolutionError",
    "DurableToolInvocationWiringBindingResolver",
]
