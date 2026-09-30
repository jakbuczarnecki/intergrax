# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-3 application plugin wiring."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI

from intergrax.applications._shared.fastapi_lifespan import LifespanFn, apply_lifespans
from intergrax.contracts.host_orchestration_application_wiring_target import (
    HostOrchestrationPluginBootstrapTarget,
)
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
    orchestration_host: HostOrchestrationPluginBootstrapTarget,
) -> PluginBootstrapResult:
    """Wire runtime plugins against a composed orchestration host."""
    return bootstrap_runtime_plugins(
        plugins,
        event_bus=orchestration_host.event_bus,
        hook_registry=orchestration_host.middleware.hooks,
        policy_engine=orchestration_host.policy_engine,
    )


def make_plugin_shutdown_lifespan(callbacks: List[Callable[[], None]]) -> LifespanFn:
    """Lifespan that runs plugin shutdown callbacks on application teardown."""

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            for callback in callbacks:
                callback()

    return _lifespan


def attach_plugin_shutdown(app: FastAPI, callbacks: List[Callable[[], None]]) -> None:
    """Register plugin shutdown hooks on a FastAPI app."""
    if not callbacks:
        return
    apply_lifespans(app, make_plugin_shutdown_lifespan(callbacks))
