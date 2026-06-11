# © Artur Czarnecki. All rights reserved.

"""Mount Tier-3 :class:`ApplicationHost` on Nexus middleware (APP-CON-1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intergrax.runtime.middleware.base import RuntimeMiddleware

if TYPE_CHECKING:
    from intergrax.harness.application_host import ApplicationHost
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.nexus.nexus_loop import NexusLoop


def _attach_middleware(nexus: NexusLoop, middleware: RuntimeMiddleware) -> None:
    pipeline = nexus._middleware  # noqa: SLF001 — Tier-3 composition hook
    if not isinstance(pipeline, MiddlewarePipeline):
        return
    existing = list(pipeline._middleware)  # noqa: SLF001
    if any(mw.name == middleware.name for mw in existing):
        return
    pipeline._middleware = sorted(  # noqa: SLF001
        [*existing, middleware],
        key=lambda item: item.priority,
    )


def apply_application_host_wiring(
    nexus: NexusLoop,
    host: ApplicationHost | None,
) -> None:
    """Attach ``ApplicationHost`` middleware when a host implementation is provided."""
    if host is None:
        return
    from intergrax.harness.hooks import ApplicationHostMiddleware

    _attach_middleware(nexus, ApplicationHostMiddleware(host))
