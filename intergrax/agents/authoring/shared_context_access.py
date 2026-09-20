# © Artur Czarnecki. All rights reserved.

"""Neutral shared-context projections for ACP runs (no Nexus backing types)."""

from __future__ import annotations

from typing import Any

from intergrax.contracts.shared_context import SharedContextView

_ACP_SHARED_CONTEXT_VIEW_KEY = "_acp_shared_context_view"


def view_from_task_metadata(_task_or_metadata: Any, *, task_id: str) -> SharedContextView:
    return SharedContextView(task_id=str(task_id))


def load_view(task_or_metadata: Any) -> SharedContextView | None:
    if not isinstance(task_or_metadata, dict):
        return None
    raw = task_or_metadata.get(_ACP_SHARED_CONTEXT_VIEW_KEY)
    if raw is None:
        return None
    if isinstance(raw, SharedContextView):
        return raw
    if isinstance(raw, dict):
        return SharedContextView.model_validate(raw)
    return None


def persist_view(task_or_metadata: Any, view: SharedContextView) -> None:
    if not isinstance(task_or_metadata, dict):
        return
    task_or_metadata[_ACP_SHARED_CONTEXT_VIEW_KEY] = view.model_dump(mode="json")
