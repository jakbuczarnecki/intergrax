# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-3 application plugin wiring."""

from __future__ import annotations

from typing import Callable, List

from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.plugins.bootstrap import PluginBootstrapResult, bootstrap_runtime_plugins
from intergrax.runtime.plugins.contract import RuntimePlugin

__all__ = [
    "PluginBootstrapResult",
    "attach_plugin_shutdown",
    "bootstrap_application_plugins",
]


def bootstrap_application_plugins(
    plugins: List[RuntimePlugin],
    *,
    nexus_loop: NexusLoop,
) -> PluginBootstrapResult:
    """Wire runtime plugins against a composed NexusLoop instance."""
    return bootstrap_runtime_plugins(
        plugins,
        event_bus=nexus_loop.event_bus,
        hook_registry=nexus_loop.middleware.hooks,
        policy_engine=nexus_loop.policy_engine,
    )


def attach_plugin_shutdown(app, callbacks: List[Callable[[], None]]) -> None:
    """Register plugin shutdown hooks on a FastAPI app."""
    if not callbacks:
        return

    @app.on_event("shutdown")
    async def _shutdown_plugins() -> None:
        for callback in callbacks:
            callback()
