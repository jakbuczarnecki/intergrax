# © Artur Czarnecki. All rights reserved.

"""Host-injected session hooks for ``Agent.run()`` (optional Tier-3 wiring)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from intergrax.agents.authoring.acp_session_host import ACPSessionHostContext
from intergrax.agents.authoring.shared_context_access import (
    load_view as neutral_load_view,
    persist_view as neutral_persist_view,
    view_from_task_metadata as neutral_view_from_task,
)


@dataclass(frozen=True, slots=True)
class AcpRuntimeSessionHooks:
    apply_runtime_profile_kernel_wiring: Callable[..., Any] | None = None
    attach_acp_catalog_exec_ctx: Callable[..., None] | None = None
    close_acp_catalog_exec_ctx: Callable[..., None] | None = None
    on_llm_routing_evaluated: Callable[..., None] | None = None
    load_shared_context_view: Callable[..., Any] | None = None
    persist_shared_context_view: Callable[..., None] | None = None
    view_shared_context_for_task: Callable[..., Any] | None = None


def default_acp_runtime_session_hooks() -> AcpRuntimeSessionHooks:
    return AcpRuntimeSessionHooks(
        load_shared_context_view=neutral_load_view,
        persist_shared_context_view=neutral_persist_view,
        view_shared_context_for_task=neutral_view_from_task,
    )


def resolve_acp_runtime_session_hooks(host: ACPSessionHostContext | None) -> AcpRuntimeSessionHooks:
    if host is not None and host.runtime_session_hooks is not None:
        return host.runtime_session_hooks
    return default_acp_runtime_session_hooks()
