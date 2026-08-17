# © Artur Czarnecki. All rights reserved.

"""Private LKW FastAPI/Uvicorn HostedApplicationRuntime adapter (APP-HOST-8A/8B)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Protocol, runtime_checkable

import uvicorn
from fastapi import FastAPI

from intergrax.applications._shared.production_agent_platform_runtime import (
    ProductionAgentPlatformRuntime,
    build_production_agent_platform_runtime,
)
from intergrax.applications._shared.production_host_composition import (
    bootstrap_production_registry_projection,
)
from intergrax.hosting.contracts.context import HostedApplicationContext
from intergrax.hosting.errors import HostedApplicationRuntimeError
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.factory import create_local_workspace_backend_app
from local_workspace_application.host.readiness import LocalWorkspaceReadinessProvider
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.hosting.readiness import _HostedLocalWorkspaceReadiness
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST

_STARTUP_POLL_INTERVAL_SECONDS = 0.01


@runtime_checkable
class _HostedServer(Protocol):
    started: bool
    should_exit: bool

    async def serve(self) -> None: ...


ApplicationFactory = Callable[
    [
        LocalWorkspaceBackendSettings,
        LocalWorkspaceReadinessProvider,
    ],
    FastAPI,
]
ServerFactory = Callable[[FastAPI, str, int], _HostedServer]


class _HostedServerProcessExit(HostedApplicationRuntimeError):
    """Normalized process-exit request raised by the embedded server."""


async def _serve_hosted_server(server: _HostedServer) -> None:
    try:
        await server.serve()
    except SystemExit as exc:
        raise _HostedServerProcessExit(
            "local workspace hosted server requested process exit"
        ) from exc


class _HostedUvicornServer(uvicorn.Server):
    """Uvicorn server that does not install process signal handlers."""

    @contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


def _default_application_factory(
    settings: LocalWorkspaceBackendSettings,
    host_readiness: LocalWorkspaceReadinessProvider,
    *,
    agent_platform_runtime: ProductionAgentPlatformRuntime | None = None,
) -> FastAPI:
    env = build_local_workspace_environment_profile(settings)
    platform_runtime = agent_platform_runtime or build_production_agent_platform_runtime()
    return create_local_workspace_backend_app(
        registry_projection=bootstrap_production_registry_projection(
            application_id=LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id,
            application_environment_id=env.profile_id,
            stores=platform_runtime.stores,
        ),
        settings=settings,
        host_readiness=host_readiness,
    )


def _default_server_factory(app: FastAPI, bind_host: str, bind_port: int) -> _HostedServer:
    config = uvicorn.Config(
        app=app,
        host=bind_host,
        port=bind_port,
        reload=False,
        workers=1,
    )
    return _HostedUvicornServer(config)


class _LocalWorkspaceHostedRuntime:
    """Private HostedApplicationRuntime wrapping LKW FastAPI + async Uvicorn serve."""

    def __init__(
        self,
        *,
        hosted_context: HostedApplicationContext,
        settings: LocalWorkspaceBackendSettings,
        bind_host: str,
        bind_port: int,
        application_factory: ApplicationFactory | None = None,
        server_factory: ServerFactory | None = None,
    ) -> None:
        self._hosted_context = hosted_context
        self._settings = settings
        self._bind_host = bind_host
        self._bind_port = bind_port
        self._application_factory = application_factory or _default_application_factory
        self._server_factory = server_factory or _default_server_factory
        self._app: FastAPI | None = None
        self._server: _HostedServer | None = None
        self._serve_task: asyncio.Task[None] | None = None
        self._start_called = False
        self._started = False

    def _clear_references(self) -> None:
        self._app = None
        self._server = None
        self._serve_task = None
        self._started = False

    async def start(self, context: HostedApplicationContext) -> None:
        if self._start_called:
            raise HostedApplicationRuntimeError(
                "local workspace hosted runtime already started"
            )
        self._start_called = True

        host_readiness = _HostedLocalWorkspaceReadiness(self._hosted_context)
        app = self._application_factory(
            self._settings,
            host_readiness,
        )

        server = self._server_factory(app, self._bind_host, self._bind_port)
        self._app = app
        self._server = server

        serve_task = asyncio.create_task(
            _serve_hosted_server(server),
            name="lkw-hosted-uvicorn-serve",
        )
        self._serve_task = serve_task

        startup_timeout_seconds = context.profile.lifecycle.default_blocking_hook_timeout_seconds
        try:
            async with asyncio.timeout(startup_timeout_seconds):
                while not server.started:
                    if serve_task.done():
                        self._raise_premature_startup_failure(serve_task)
                    await asyncio.sleep(_STARTUP_POLL_INTERVAL_SECONDS)
        except TimeoutError as exc:
            server.should_exit = True
            if not serve_task.done():
                serve_task.cancel()
            await asyncio.gather(serve_task, return_exceptions=True)
            self._clear_references()
            raise HostedApplicationRuntimeError(
                "local workspace hosted server startup timed out"
            ) from exc

        self._started = True

    def _raise_premature_startup_failure(self, serve_task: asyncio.Task[None]) -> None:
        try:
            exception = serve_task.exception()
        except asyncio.CancelledError as exc:
            self._clear_references()
            raise HostedApplicationRuntimeError(
                "local workspace hosted server failed before startup"
            ) from exc
        self._clear_references()
        if exception is not None:
            cause: BaseException = exception
            if isinstance(exception, _HostedServerProcessExit) and exception.__cause__ is not None:
                cause = exception.__cause__
            raise HostedApplicationRuntimeError(
                "local workspace hosted server failed before startup"
            ) from cause
        raise HostedApplicationRuntimeError(
            "local workspace hosted server exited before startup"
        )

    async def ready(self, context: HostedApplicationContext) -> bool:
        del context  # platform READY is owned by HostedApplicationReadinessService
        return (
            self._started
            and self._app is not None
            and self._server is not None
            and self._serve_task is not None
            and not self._serve_task.done()
            and self._server.started
        )

    async def stop(self, context: HostedApplicationContext) -> None:
        del context  # platform engine owns shutdown budget
        if self._app is None and self._serve_task is None:
            return

        server = self._server
        serve_task = self._serve_task
        try:
            if server is not None:
                server.should_exit = True
            if serve_task is not None:
                await serve_task
        finally:
            self._clear_references()
