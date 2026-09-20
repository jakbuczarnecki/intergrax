# © Artur Czarnecki. All rights reserved.

"""Host-injected session hooks for ``Agent.run()`` (optional Tier-3 wiring)."""

from __future__ import annotations

from typing import Protocol

from intergrax.agents.authoring.acp_runtime_session_ports import AcpRuntimeSessionHooks
from intergrax.agents.authoring.shared_context_access import (
    load_view as neutral_load_view,
    persist_view as neutral_persist_view,
    view_from_task_metadata as neutral_view_from_task,
)

__all__ = [
    "AcpRuntimeSessionHooks",
    "default_acp_runtime_session_hooks",
    "resolve_acp_runtime_session_hooks",
]


class _HostRuntimeSessionHooks(Protocol):
    runtime_session_hooks: AcpRuntimeSessionHooks | None


def default_acp_runtime_session_hooks() -> AcpRuntimeSessionHooks:
    return AcpRuntimeSessionHooks(
        load_shared_context_view=neutral_load_view,
        persist_shared_context_view=neutral_persist_view,
        view_shared_context_for_task=neutral_view_from_task,
    )


def resolve_acp_runtime_session_hooks(host: _HostRuntimeSessionHooks | None) -> AcpRuntimeSessionHooks:
    if host is not None and host.runtime_session_hooks is not None:
        return host.runtime_session_hooks
    return default_acp_runtime_session_hooks()
