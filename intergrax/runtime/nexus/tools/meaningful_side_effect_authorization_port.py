# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Nexus seam — canonical meaningful side-effect authorization (contract surface only)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.collaborative_work import CollaborativeWorkEnforcementRequest


@runtime_checkable
class MeaningfulSideEffectAuthorizationPort(Protocol):
    """Authorize consequential tool effects before physical execution."""

    def authorize(
        self,
        request: CollaborativeWorkEnforcementRequest,
        *,
        source_agent_id: str = "platform.orchestration.tool_invocation",
        source_step_id: str | None = None,
    ) -> object:
        """Return ``MeaningfulSideEffectAuthorizationResult`` (runtime policy type)."""


__all__ = ["MeaningfulSideEffectAuthorizationPort"]
