# © Artur Czarnecki. All rights reserved.

"""Nexus-backed ``SharedContextAccessPort`` (runtime implementation detail)."""

from __future__ import annotations

from intergrax.contracts.agent_run import AgentRunRequest
from intergrax.contracts.shared_context import SharedContextView
from intergrax.contracts.shared_context_access import SharedContextAccessPort
from intergrax.runtime.nexus.agents.shared_context_bridge import (
    load_view,
    persist_view,
    view_from_task_metadata,
)


class NexusSharedContextAccess:
    """Bind shared-context bridge operations to a single ACP run request."""

    __slots__ = ("_request",)

    def __init__(self, request: AgentRunRequest) -> None:
        self._request = request

    def load(self) -> SharedContextView | None:
        return load_view(self._request.metadata)

    def persist(self, view: SharedContextView) -> None:
        persist_view(self._request.metadata, view)

    def project(self, *, task_id: str) -> SharedContextView:
        return view_from_task_metadata(self._request.metadata, task_id=task_id)


def nexus_shared_context_access_for_run(request: AgentRunRequest) -> SharedContextAccessPort:
    return NexusSharedContextAccess(request)
