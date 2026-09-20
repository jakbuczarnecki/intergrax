# © Artur Czarnecki. All rights reserved.

"""Host-injected session hooks for ``Agent.run()`` (optional Tier-3 wiring)."""

from __future__ import annotations

from typing import Protocol

from intergrax.agents.authoring.acp_runtime_session_ports import AcpRuntimeSessionHooks
from intergrax.agents.authoring.shared_context_access import neutral_shared_context_access_for_run

__all__ = [
    "AcpRuntimeSessionHooks",
    "default_acp_runtime_session_hooks",
    "resolve_acp_runtime_session_hooks",
]


class _HostRuntimeSessionHooks(Protocol):
    runtime_session_hooks: AcpRuntimeSessionHooks | None


def default_acp_runtime_session_hooks() -> AcpRuntimeSessionHooks:
    return AcpRuntimeSessionHooks(
        resolve_shared_context_access=neutral_shared_context_access_for_run,
    )


def resolve_acp_runtime_session_hooks(host: _HostRuntimeSessionHooks | None) -> AcpRuntimeSessionHooks:
    if host is not None and host.runtime_session_hooks is not None:
        return host.runtime_session_hooks
    return default_acp_runtime_session_hooks()
