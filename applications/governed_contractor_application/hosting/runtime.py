# © Artur Czarnecki. All rights reserved.

"""Private GCA FastAPI/Uvicorn HostedApplicationRuntime adapter."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from typing import Protocol, runtime_checkable

import uvicorn
from fastapi import FastAPI
from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionServingSession,
)
from intergrax.hosting.contracts.context import HostedApplicationContext
from intergrax.hosting.errors import HostedApplicationRuntimeError

from governed_contractor_application.host.governed_contractor_host_runtime_composition import (
    GovernedContractorHostRuntimeComposition,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.host.strict_production_serving import (
    create_governed_contractor_strict_production_asgi_app,
)

_STARTUP_POLL_INTERVAL_SECONDS = 0.01


@runtime_checkable
class _HostedServer(Protocol):
    started: bool
    should_exit: bool

    async def serve(self) -> None: ...


ApplicationFactory = Callable[[], FastAPI]
ServerFactory = Callable[[FastAPI, str, int], _HostedServer]


class _HostedServerProcessExit(HostedApplicationRuntimeError):
    """Normalized process-exit request raised by the embedded server."""


async def _serve_hosted_server(server: _HostedServer) -> None:
    try:
        await server.serve()
    except SystemExit as exc:
        raise _HostedServerProcessExit(
            "governed contractor hosted server requested process exit",
        ) from exc


class _HostedUvicornServer(uvicorn.Server):
    """Uvicorn server that does not install process signal handlers."""

    @contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


def _default_application_factory(
    session: StrictProductionServingSession,
    settings: GovernedContractorBackendSettings,
    host_runtime: GovernedContractorHostRuntimeComposition | None,
) -> FastAPI:
    return create_governed_contractor_strict_production_asgi_app(
        session,
        settings=settings,
        host_runtime=host_runtime,
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


class _GovernedContractorHostedRuntime:
    """Private HostedApplicationRuntime wrapping GCA strict FastAPI + async Uvicorn serve."""

    def __init__(
        self,
        *,
        hosted_context: HostedApplicationContext,
        session: StrictProductionServingSession,
        settings: GovernedContractorBackendSettings,
        bind_host: str,
        bind_port: int,
        host_runtime: GovernedContractorHostRuntimeComposition | None = None,
        application_factory: ApplicationFactory | None = None,
        server_factory: ServerFactory | None = None,
    ) -> None:
        self._hosted_context = hosted_context
        self._session = session
        self._settings = settings
        self._bind_host = bind_host
        self._bind_port = bind_port
        self._host_runtime = host_runtime
        self._application_factory = application_factory or partial(
            _default_application_factory,
            session,
            settings,
            host_runtime,
        )
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
                "governed contractor hosted runtime already started",
            )
        self._start_called = True

        app = self._application_factory()
        server = self._server_factory(app, self._bind_host, self._bind_port)
        self._app = app
        self._server = server

        serve_task = asyncio.create_task(
            _serve_hosted_server(server),
            name="gca-hosted-uvicorn-serve",
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
                "governed contractor hosted server startup timed out",
            ) from exc

        self._started = True

    def _raise_premature_startup_failure(self, serve_task: asyncio.Task[None]) -> None:
        try:
            exception = serve_task.exception()
        except asyncio.CancelledError as exc:
            self._clear_references()
            raise HostedApplicationRuntimeError(
                "governed contractor hosted server failed before startup",
            ) from exc
        self._clear_references()
        if exception is not None:
            cause: BaseException = exception
            if isinstance(exception, _HostedServerProcessExit) and exception.__cause__ is not None:
                cause = exception.__cause__
            raise HostedApplicationRuntimeError(
                "governed contractor hosted server failed before startup",
            ) from cause
        raise HostedApplicationRuntimeError(
            "governed contractor hosted server exited before startup",
        )

    async def ready(self, context: HostedApplicationContext) -> bool:
        del context
        return (
            self._started
            and self._app is not None
            and self._server is not None
            and self._serve_task is not None
            and not self._serve_task.done()
            and self._server.started
        )

    async def stop(self, context: HostedApplicationContext) -> None:
        del context
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
