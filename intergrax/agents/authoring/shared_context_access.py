# © Artur Czarnecki. All rights reserved.

"""Neutral shared-context access for ACP runs (no Nexus backing types)."""

from __future__ import annotations

from intergrax.contracts.agent_run import AgentRunRequest
from intergrax.contracts.shared_context import SharedContextView
from intergrax.contracts.shared_context_access import SharedContextAccessPort

_ACP_SHARED_CONTEXT_VIEW_KEY = "_acp_shared_context_view"


class InMemorySharedContextAccess:
    """Ephemeral store for tests and runs without metadata persistence."""

    __slots__ = ("_stored",)

    def __init__(self) -> None:
        self._stored: SharedContextView | None = None

    def load(self) -> SharedContextView | None:
        return self._stored

    def persist(self, view: SharedContextView) -> None:
        self._stored = view

    def project(self, *, task_id: str) -> SharedContextView:
        return SharedContextView(task_id=str(task_id))


class RequestBoundSharedContextAccess:
    """Default neutral backing: ACP metadata slot on ``AgentRunRequest``."""

    __slots__ = ("_request",)

    def __init__(self, request: AgentRunRequest) -> None:
        self._request = request

    def load(self) -> SharedContextView | None:
        raw = self._request.metadata.get(_ACP_SHARED_CONTEXT_VIEW_KEY)
        if raw is None:
            return None
        if isinstance(raw, SharedContextView):
            return raw
        if isinstance(raw, dict):
            return SharedContextView.model_validate(raw)
        msg = (
            "stored shared-context view must be SharedContextView or mapping; "
            f"got {type(raw).__name__}"
        )
        raise TypeError(msg)

    def persist(self, view: SharedContextView) -> None:
        self._request.metadata[_ACP_SHARED_CONTEXT_VIEW_KEY] = view.model_dump(mode="json")

    def project(self, *, task_id: str) -> SharedContextView:
        return SharedContextView(task_id=str(task_id))


def neutral_shared_context_access_for_run(request: AgentRunRequest) -> SharedContextAccessPort:
    return RequestBoundSharedContextAccess(request)
