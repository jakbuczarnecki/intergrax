# © Artur Czarnecki. All rights reserved.

"""Neutral shared-context projections for ACP runs (no Nexus backing types)."""

from __future__ import annotations

from collections.abc import MutableMapping

from intergrax.contracts.shared_context import SharedContextView

AcpSharedContextMetadata = MutableMapping[str, object]

_ACP_SHARED_CONTEXT_VIEW_KEY = "_acp_shared_context_view"


def _require_acp_metadata(carrier: object) -> AcpSharedContextMetadata:
    if not isinstance(carrier, MutableMapping):
        msg = (
            "shared-context metadata carrier must be a mutable mapping; "
            f"got {type(carrier).__name__}"
        )
        raise TypeError(msg)
    return carrier


def view_from_task_metadata(_metadata: AcpSharedContextMetadata, *, task_id: str) -> SharedContextView:
    return SharedContextView(task_id=str(task_id))


def load_view(metadata: AcpSharedContextMetadata) -> SharedContextView | None:
    meta = _require_acp_metadata(metadata)
    raw = meta.get(_ACP_SHARED_CONTEXT_VIEW_KEY)
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


def persist_view(metadata: AcpSharedContextMetadata, view: SharedContextView) -> None:
    meta = _require_acp_metadata(metadata)
    meta[_ACP_SHARED_CONTEXT_VIEW_KEY] = view.model_dump(mode="json")
