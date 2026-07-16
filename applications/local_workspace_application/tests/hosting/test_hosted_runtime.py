# © Artur Czarnecki. All rights reserved.

"""APP-HOST-8A — LKW hosted runtime adapter tests."""

from __future__ import annotations

import ast
import asyncio
import inspect
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI

from intergrax.hosting.contracts.context import (
    HostedApplicationContext,
    HostedApplicationPaths,
    HostedApplicationProcessIdentity,
)
from intergrax.hosting.contracts.events import HostedApplicationEvent
from intergrax.hosting.contracts.lifecycle import (
    HostedApplicationLifecycleSnapshot,
    HostedApplicationLifecycleState,
    HostedApplicationShutdownCoordinator,
)
from intergrax.hosting.contracts.policies import LifecyclePolicy
from intergrax.hosting.errors import HostedApplicationRuntimeError
from intergrax.hosting.services import HostedApplicationServiceRegistry
from local_workspace_application.host.lifecycle import LocalWorkspaceHostLifecycle
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.hosting import build_local_workspace_hosted_profile
from local_workspace_application.hosting.runtime import (
    _HostedUvicornServer,
    _LocalWorkspaceHostedRuntime,
)


class _Clock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class _Logger:
    def debug(self, message: str, **fields: object) -> None:
        del message, fields

    def info(self, message: str, **fields: object) -> None:
        del message, fields

    def warning(self, message: str, **fields: object) -> None:
        del message, fields

    def error(self, message: str, **fields: object) -> None:
        del message, fields


class _EventPublisher:
    async def publish(self, event: HostedApplicationEvent) -> None:
        del event


class _Shutdown:
    def request_shutdown(self, *, reason_code: str = "test") -> None:
        del reason_code

    def is_shutdown_requested(self) -> bool:
        return False

    async def wait_until_requested(self) -> None:
        return None


class _LifecycleProvider:
    def snapshot(self) -> HostedApplicationLifecycleSnapshot:
        return HostedApplicationLifecycleSnapshot(
            state=HostedApplicationLifecycleState.READY,
            accepting_new_work=True,
            shutdown_requested=False,
            last_transition_at=datetime.now(timezone.utc),
            reason_code="ready",
        )


def _context_for_runtime(
    *,
    startup_timeout_seconds: float = 2.0,
) -> HostedApplicationContext:
    profile = build_local_workspace_hosted_profile(
        settings=LocalWorkspaceBackendSettings(),
    )
    # Rebuild public view with a short startup bound for timeout tests.
    public = profile.public_view().model_copy(
        update={"lifecycle": LifecyclePolicy(default_blocking_hook_timeout_seconds=startup_timeout_seconds)},
    )
    return HostedApplicationContext(
        application_id=profile.application_id,
        instance_id="01TESTHOSTEDRUNTIMEINSTANCE00001",
        profile=public,
        profile_digest=profile.profile_digest(),
        paths=HostedApplicationPaths(
            data_home=Path("build/test-lkw-hosting-8a-runtime"),
            run_directory=Path("build/test-lkw-hosting-8a-runtime/run"),
        ),
        process_identity=HostedApplicationProcessIdentity(
            process_id=1,
            started_at=datetime.now(timezone.utc),
        ),
        services=HostedApplicationServiceRegistry(),
        clock=_Clock(),
        logger=_Logger(),
        event_publisher=_EventPublisher(),
        shutdown=cast(HostedApplicationShutdownCoordinator, _Shutdown()),
        lifecycle=_LifecycleProvider(),
    )


def _ready_lifecycle() -> LocalWorkspaceHostLifecycle:
    lifecycle = LocalWorkspaceHostLifecycle()
    lifecycle.set_executor_available(True)
    lifecycle.transition_to_ready()
    return lifecycle


def _fake_app_factory(
    settings: LocalWorkspaceBackendSettings,
    *,
    lifecycle: LocalWorkspaceHostLifecycle | None = None,
    calls: list[LocalWorkspaceBackendSettings] | None = None,
) -> FastAPI:
    if calls is not None:
        calls.append(settings)
    app = FastAPI()
    app.state.lkw_host_lifecycle = lifecycle or _ready_lifecycle()
    return app


class _FakeServer:
    def __init__(
        self,
        *,
        failure_before_startup: BaseException | None = None,
        exit_before_startup: bool = False,
        failure_after_startup: BaseException | None = None,
        never_start: bool = False,
    ) -> None:
        self.started = False
        self.should_exit = False
        self.serve_started = asyncio.Event()
        self.allow_start = asyncio.Event()
        self.serve_finished = asyncio.Event()
        self._failure_before_startup = failure_before_startup
        self._exit_before_startup = exit_before_startup
        self._failure_after_startup = failure_after_startup
        self._never_start = never_start

    async def serve(self) -> None:
        self.serve_started.set()
        try:
            if self._failure_before_startup is not None:
                raise self._failure_before_startup
            if self._exit_before_startup:
                return
            if self._never_start:
                while not self.should_exit:
                    await asyncio.sleep(0.01)
                return
            await self.allow_start.wait()
            self.started = True
            while not self.should_exit:
                if self._failure_after_startup is not None:
                    raise self._failure_after_startup
                await asyncio.sleep(0.01)
        finally:
            self.serve_finished.set()


@pytest.mark.asyncio
async def test_ready_false_before_start() -> None:
    settings = LocalWorkspaceBackendSettings()
    runtime = _LocalWorkspaceHostedRuntime(
        settings=settings,
        bind_host="127.0.0.1",
        bind_port=8020,
        application_factory=lambda s: _fake_app_factory(s),
        server_factory=lambda app, host, port: _FakeServer(),
    )
    context = _context_for_runtime()
    assert await runtime.ready(context) is False


@pytest.mark.asyncio
async def test_start_ready_stop_lifecycle() -> None:
    settings = LocalWorkspaceBackendSettings()
    app_calls: list[LocalWorkspaceBackendSettings] = []
    servers: list[_FakeServer] = []
    lifecycle = _ready_lifecycle()

    def application_factory(resolved: LocalWorkspaceBackendSettings) -> FastAPI:
        return _fake_app_factory(resolved, lifecycle=lifecycle, calls=app_calls)

    def server_factory(app: FastAPI, host: str, port: int) -> _FakeServer:
        del app, host, port
        server = _FakeServer()
        servers.append(server)
        server.allow_start.set()
        return server

    runtime = _LocalWorkspaceHostedRuntime(
        settings=settings,
        bind_host="127.0.0.1",
        bind_port=8020,
        application_factory=application_factory,
        server_factory=server_factory,
    )
    context = _context_for_runtime()

    await runtime.start(context)
    assert len(app_calls) == 1
    assert len(servers) == 1
    assert runtime._serve_task is not None  # noqa: SLF001
    assert servers[0].started is True
    assert await runtime.ready(context) is True

    lifecycle.set_executor_available(False)
    assert await runtime.ready(context) is False
    lifecycle.set_executor_available(True)
    assert await runtime.ready(context) is True

    stop_task = asyncio.create_task(runtime.stop(context))
    await asyncio.wait_for(servers[0].serve_finished.wait(), timeout=2.0)
    await stop_task
    assert servers[0].should_exit is True
    assert runtime._app is None  # noqa: SLF001
    assert runtime._server is None  # noqa: SLF001
    assert runtime._serve_task is None  # noqa: SLF001
    assert runtime._lifecycle is None  # noqa: SLF001

    await runtime.stop(context)  # idempotent


@pytest.mark.asyncio
async def test_ready_false_when_serve_task_finishes() -> None:
    settings = LocalWorkspaceBackendSettings()
    server = _FakeServer()
    server.allow_start.set()

    runtime = _LocalWorkspaceHostedRuntime(
        settings=settings,
        bind_host="127.0.0.1",
        bind_port=8020,
        application_factory=lambda s: _fake_app_factory(s),
        server_factory=lambda app, host, port: server,
    )
    context = _context_for_runtime()
    await runtime.start(context)
    assert await runtime.ready(context) is True
    server.should_exit = True
    await asyncio.wait_for(server.serve_finished.wait(), timeout=2.0)
    assert await runtime.ready(context) is False
    await runtime.stop(context)


@pytest.mark.asyncio
async def test_second_start_rejected() -> None:
    settings = LocalWorkspaceBackendSettings()
    server = _FakeServer()
    server.allow_start.set()
    runtime = _LocalWorkspaceHostedRuntime(
        settings=settings,
        bind_host="127.0.0.1",
        bind_port=8020,
        application_factory=lambda s: _fake_app_factory(s),
        server_factory=lambda app, host, port: server,
    )
    context = _context_for_runtime()
    await runtime.start(context)
    with pytest.raises(HostedApplicationRuntimeError, match="already started"):
        await runtime.start(context)
    await runtime.stop(context)


@pytest.mark.asyncio
async def test_app_factory_failure_creates_no_server_or_task() -> None:
    settings = LocalWorkspaceBackendSettings()
    servers: list[_FakeServer] = []

    def application_factory(_settings: LocalWorkspaceBackendSettings) -> FastAPI:
        raise RuntimeError("app-factory-boom")

    def server_factory(app: FastAPI, host: str, port: int) -> _FakeServer:
        del app, host, port
        server = _FakeServer()
        servers.append(server)
        return server

    runtime = _LocalWorkspaceHostedRuntime(
        settings=settings,
        bind_host="127.0.0.1",
        bind_port=8020,
        application_factory=application_factory,
        server_factory=server_factory,
    )
    context = _context_for_runtime()
    with pytest.raises(RuntimeError, match="app-factory-boom"):
        await runtime.start(context)
    assert servers == []
    assert runtime._serve_task is None  # noqa: SLF001
    assert runtime._server is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_serve_raises_before_started() -> None:
    settings = LocalWorkspaceBackendSettings()
    cause = RuntimeError("serve-boom")
    server = _FakeServer(failure_before_startup=cause)
    runtime = _LocalWorkspaceHostedRuntime(
        settings=settings,
        bind_host="127.0.0.1",
        bind_port=8020,
        application_factory=lambda s: _fake_app_factory(s),
        server_factory=lambda app, host, port: server,
    )
    context = _context_for_runtime()
    with pytest.raises(HostedApplicationRuntimeError, match="failed before startup") as exc_info:
        await runtime.start(context)
    assert exc_info.value.__cause__ is cause


@pytest.mark.asyncio
async def test_system_exit_before_startup_is_normalized() -> None:
    settings = LocalWorkspaceBackendSettings()
    cause = SystemExit(1)
    server = _FakeServer(failure_before_startup=cause)
    runtime = _LocalWorkspaceHostedRuntime(
        settings=settings,
        bind_host="127.0.0.1",
        bind_port=8020,
        application_factory=lambda s: _fake_app_factory(s),
        server_factory=lambda app, host, port: server,
    )
    context = _context_for_runtime()
    with pytest.raises(HostedApplicationRuntimeError, match="failed before startup") as exc_info:
        await runtime.start(context)
    assert isinstance(exc_info.value.__cause__, SystemExit)
    assert exc_info.value.__cause__ is cause
    assert runtime._app is None  # noqa: SLF001
    assert runtime._server is None  # noqa: SLF001
    assert runtime._serve_task is None  # noqa: SLF001
    assert runtime._lifecycle is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_system_exit_after_startup_does_not_exit_process() -> None:
    settings = LocalWorkspaceBackendSettings()
    cause = SystemExit(1)
    server = _FakeServer()
    server.allow_start.set()
    runtime = _LocalWorkspaceHostedRuntime(
        settings=settings,
        bind_host="127.0.0.1",
        bind_port=8020,
        application_factory=lambda s: _fake_app_factory(s),
        server_factory=lambda app, host, port: server,
    )
    context = _context_for_runtime()
    await runtime.start(context)
    assert await runtime.ready(context) is True
    server._failure_after_startup = cause  # noqa: SLF001
    await asyncio.wait_for(server.serve_finished.wait(), timeout=2.0)
    assert await runtime.ready(context) is False
    with pytest.raises(HostedApplicationRuntimeError) as exc_info:
        await runtime.stop(context)
    assert isinstance(exc_info.value, HostedApplicationRuntimeError)
    chain: list[BaseException] = []
    current: BaseException | None = exc_info.value
    while current is not None:
        chain.append(current)
        current = current.__cause__
    assert any(isinstance(item, SystemExit) for item in chain)
    assert cause in chain
    assert runtime._app is None  # noqa: SLF001
    assert runtime._server is None  # noqa: SLF001
    assert runtime._serve_task is None  # noqa: SLF001
    assert runtime._lifecycle is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_serve_exits_before_started() -> None:
    settings = LocalWorkspaceBackendSettings()
    server = _FakeServer(exit_before_startup=True)
    runtime = _LocalWorkspaceHostedRuntime(
        settings=settings,
        bind_host="127.0.0.1",
        bind_port=8020,
        application_factory=lambda s: _fake_app_factory(s),
        server_factory=lambda app, host, port: server,
    )
    context = _context_for_runtime()
    with pytest.raises(HostedApplicationRuntimeError, match="exited before startup"):
        await runtime.start(context)


@pytest.mark.asyncio
async def test_startup_timeout_cancels_serve_task() -> None:
    settings = LocalWorkspaceBackendSettings()
    server = _FakeServer(never_start=True)
    runtime = _LocalWorkspaceHostedRuntime(
        settings=settings,
        bind_host="127.0.0.1",
        bind_port=8020,
        application_factory=lambda s: _fake_app_factory(s),
        server_factory=lambda app, host, port: server,
    )
    context = _context_for_runtime(startup_timeout_seconds=0.05)
    with pytest.raises(HostedApplicationRuntimeError, match="startup timed out"):
        await runtime.start(context)
    await asyncio.wait_for(server.serve_finished.wait(), timeout=2.0)
    assert server.should_exit is True
    assert runtime._serve_task is None  # noqa: SLF001


def test_runtime_module_signal_ownership_boundary() -> None:
    runtime_path = (
        Path(__file__).resolve().parents[2] / "hosting" / "runtime.py"
    )
    source = runtime_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_names.add(node.module.split(".")[0])
            for alias in node.names:
                imported_names.add(alias.name)
    assert "signal" not in imported_names
    assert "PortableForegroundSignalAdapter" not in imported_names
    assert "HostedApplicationControlCoordinator" not in imported_names

    # Production server suppresses Uvicorn signal capture.
    source_cm = inspect.getsource(_HostedUvicornServer.capture_signals)
    assert "yield" in source_cm
    assert "signal.signal" not in source_cm
    assert "import signal" not in source
