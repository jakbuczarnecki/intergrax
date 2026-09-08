# © Artur Czarnecki. All rights reserved.

"""Governance port for tool invocation integration."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.agent_runtime_governance import (
    ToolAuthorizationDecision,
    ToolAuthorizationRequest,
)


@runtime_checkable
class AgentRuntimeGovernancePort(Protocol):
    """
    Pre-execution governance boundary evaluated by RuntimeToolInvoker.

    Implementations must be pure evaluation — no tool execution side effects.
    """

    def authorize_tool(
        self,
        request: ToolAuthorizationRequest,
    ) -> ToolAuthorizationDecision: ...
