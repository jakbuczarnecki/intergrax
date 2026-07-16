# © Artur Czarnecki. All rights reserved.

"""Private LKW FastAPI/Uvicorn HostedApplicationRuntime adapter (APP-HOST-8A)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Protocol, runtime_checkable

import uvicorn
from fastapi import FastAPI

from intergrax.hosting.contracts.context import HostedApplicationContext
from intergrax.hosting.errors import HostedApplicationRuntimeError
from local_workspace_application.host.factory import create_local_workspace_backend_app
from local_workspace_application.host.lifecycle import LocalWorkspaceHostLifecycle
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings

_STARTUP_POLL_INTERVAL_SECONDS = 0.01


@runtime_checkable
class _HostedServer(Protocol):
    started: bool
    should_exit: bool

    async def serve(self) -> None: ...


ApplicationFactory = Callable[[LocalWorkspaceBackendSettings], FastAPI]
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


def _default_application_factory(settings: LocalWorkspaceBackendSettings) -> FastAPI:
    return create_local_workspace_backend_app(settings=settings)


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
        settings: LocalWorkspaceBackendSettings,
        bind_host: str,
        bind_port: int,
        application_factory: ApplicationFactory | None = None,
        server_factory: ServerFactory | None = None,
    ) -> None:
        self._settings = settings
        self._bind_host = bind_host
        self._bind_port = bind_port
        self._application_factory = application_factory or _default_application_factory
        self._server_factory = server_factory or _default_server_factory
        self._app: FastAPI | None = None
        self._server: _HostedServer | None = None
        self._serve_task: asyncio.Task[None] | None = None
        self._lifecycle: LocalWorkspaceHostLifecycle | None = None
        self._start_called = False
        self._started = False

    def _clear_references(self) -> None:
        self._app = None
        self._server = None
        self._serve_task = None
        self._lifecycle = None
        self._started = False

    async def start(self, context: HostedApplicationContext) -> None:
        if self._start_called:
            raise HostedApplicationRuntimeError(
                "local workspace hosted runtime already started"
            )
        self._start_called = True

        app = self._application_factory(self._settings)
        lifecycle = getattr(app.state, "lkw_host_lifecycle", None)
        if lifecycle is not None and not isinstance(lifecycle, LocalWorkspaceHostLifecycle):
            lifecycle = None

        server = self._server_factory(app, self._bind_host, self._bind_port)
        self._app = app
        self._server = server
        self._lifecycle = lifecycle

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
        del context  # APP-HOST-8A: readiness is owned by LKW lifecycle
        if not self._started:
            return False
        if self._app is None:
            return False
        if self._server is None:
            return False
        if self._serve_task is None:
            return False
        if self._serve_task.done():
            return False
        if not self._server.started:
            return False
        if self._lifecycle is None:
            return False
        return self._lifecycle.is_ready()

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
