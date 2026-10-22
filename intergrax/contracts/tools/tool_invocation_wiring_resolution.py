# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Public per-invocation tool wiring resolver seam (catalog / qualified invocation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

_RegistrationWiringT = TypeVar("_RegistrationWiringT")
_InvocationWiringT = TypeVar("_InvocationWiringT")


@runtime_checkable
class ToolInvocationWiringResolver(Protocol[_RegistrationWiringT, _InvocationWiringT]):
    """Read-only dependency projection for a single tool invocation."""

    def resolve(
        self,
        *,
        tool_id: str,
        invocation_context: ToolInvocationContext,
        registration_wiring: _RegistrationWiringT,
    ) -> _InvocationWiringT: ...


@dataclass(frozen=True, slots=True)
class ToolInvocationContext:
    """Narrow invocation identity for wiring resolution (no arbitrary metadata bag)."""

    run_id: str
    step_id: str
    tool_id: str
    agent_id: str | None = None
    tenant_id: str | None = None
    correlation_request_id: str | None = None
    wiring_resolver: ToolInvocationWiringResolver | None = None


__all__ = [
    "ToolInvocationContext",
    "ToolInvocationWiringResolver",
]
